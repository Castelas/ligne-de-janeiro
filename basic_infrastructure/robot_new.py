# robot2.py — seguidor de linha com ROI, confiança robusta e controle PI suave
# Estados: FOLLOW / LOST com busca ativa e anti-lock
# Vídeo via PUB (tcp://*:5555) e teclas via REQ para SERVER_IP:5005

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import zmq
import base64
import time
import numpy as np
import serial

# ============================= PARÂMETROS GERAIS =============================
# --- REDE ---
SERVER_IP = "192.168.137.78"     # <--- AJUSTE para o IP do servidor/control.py
MY_ID     = "bot001"

# --- VISÃO ---
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 180

# ROI para o CONTROLE (foco no que está perto do robô)
ROI_BOTTOM_FRAC = 0.55           # usa parte inferior da imagem (>= 0.5 recomendado)

# Filtros de confiança da faixa (reduz falsos positivos)
LINE_POLARITY   = 'white'        # 'white' | 'black' | 'auto' (prefira evitar 'auto')
MIN_AREA_FRAC   = 0.004          # contornos muito pequenos são descartados
MAX_AREA_FRAC   = 0.20           # contornos muito grandes (piso) são descartados
ASPECT_MIN      = 2.2            # alongamento mínimo (comprimento/largura)
SOLIDITY_MIN    = 0.65           # compacidade do contorno (área/área do convexo)
WHITE_MEAN_MIN  = 160            # média de cinza mínima p/ linha branca
BLACK_MEAN_MAX  = 90             # média de cinza máxima p/ linha preta
USE_ADAPTIVE    = False          # threshold adaptativo desligado por padrão

# --- CONTROLE (PI com suavização do erro) ---
Kp = 0.7                         # ganho proporcional
Ki = 0.9                         # ganho integral (pequeno; use com anti-windup)
I_MAX = 220.0                    # teto do integrador (anti-windup)
I_LEAK_TAU = 2.0                 # constante de tempo (s) do “vazamento” do I (leaky integrator)
ERR_TAU = 0.15                   # (s) time-constant do filtro 1ª ordem no erro (EMA)
DEAD_BAND = 5                    # insensibilidade para erros muito pequenos (px)

# Mapeamento para velocidades
VELOCIDADE_BASE = 150
VELOCIDADE_MAX  = 255
FORWARD_MIN     = 60             # no FOLLOW, não permitir ré; mínimos para evitar parada
BASE_SCALE_MIN  = 0.35           # base mínima relativa em erro grande
E_MAX_PIX       = IMG_WIDTH // 2 # para escalonar base com |erro|

# Estados e recuperação (LOST)
LOST_MAX_FRAMES   = 6            # frames sem confiança para entrar em LOST
FOUND_MIN_FRAMES  = 2            # histerese: nº mínimo de frames confiáveis para sair do LOST
SEARCH_SPEED      = 120          # velocidade angular (girar no lugar) em LOST
LOST_SPIN_PERIOD  = 1.2          # (s) alterna direção a cada período para não travar
LOST_FORWARD      = 0            # avanço em LOST (0 = gira no lugar)
# Nota: Em FOLLOW não deixamos ré. Em LOST permitimos giro com sinais opostos.

# --- SERIAL ---
PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200

# ============================ FERRAMENTAS DE VISÃO ==========================
def threshold_pair(gray_blur):
    if USE_ADAPTIVE:
        th_w = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 31, -5)
        th_b = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 31, -5)
    else:
        _, th_w = cv2.threshold(gray_blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        _, th_b = cv2.threshold(gray_blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    return th_w, th_b

def choose_mask(th_w, th_b, gray, y0, w, h):
    """
    Seleciona contorno “plausível de faixa” na ROI inferior.
    Retorna: (contorno, cx_full, cy_full, conf)
    """
    roi_h = h - y0
    roi_area = w * roi_h

    def select_from(th, expect_white):
        roi = th[y0:h, :]
        eroded = cv2.erode(roi, None, iterations=1)
        dilated = cv2.dilate(eroded, None, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, 0

        best = None
        best_len = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < roi_area * MIN_AREA_FRAC:  # pequeno demais
                continue
            if area > roi_area * MAX_AREA_FRAC:  # grande demais (piso inteiro)
                continue

            rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
            (rw, rh) = rect[1]
            if rw < 1 or rh < 1:
                continue
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            if aspect < ASPECT_MIN:
                continue

            # Solidez (compacidade)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = float(area) / float(hull_area)
            if solidity < SOLIDITY_MIN:
                continue

            # Coerência com polaridade via média de intensidade na máscara do contorno
            mask_c = np.zeros((roi_h, w), dtype=np.uint8)
            cv2.drawContours(mask_c, [c], -1, 255, -1)
            mean_val = cv2.mean(gray[y0:h, :], mask=mask_c)[0]
            if expect_white and mean_val < WHITE_MEAN_MIN:
                continue
            if (not expect_white) and mean_val > BLACK_MEAN_MAX:
                continue

            # Mantém o “mais comprido”
            length = max(rw, rh)
            if length > best_len:
                best_len = length
                best = c

        if best is None:
            return None, None, None, 0

        # Centroide dentro da ROI
        M = cv2.moments(best)
        if M["m00"] <= 1e-6:
            return None, None, None, 0
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Para coordenadas da imagem completa
        return best, cx, cy + y0, 1

    if LINE_POLARITY == 'white':
        return select_from(th_w, True)
    elif LINE_POLARITY == 'black':
        return select_from(th_b, False)
    else:
        # auto: tenta branco; se falhar, tenta preto
        c, cx, cy, conf = select_from(th_w, True)
        if conf == 1:
            return c, cx, cy, conf
        return select_from(th_b, False)

def process_image_for_control(frame_bgr):
    """
    Retorna (frame_annot, erro_px, conf), usando só a ROI inferior.
    """
    h, w = frame_bgr.shape[:2]
    cx_img = w // 2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th_w, th_b = threshold_pair(blur)

    y0 = int(h * ROI_BOTTOM_FRAC)
    cont, cx_full, cy_full, conf = choose_mask(th_w, th_b, gray, y0, w, h)

    erro = 0
    out = frame_bgr

    # desenha a linha da ROI para debug
    cv2.line(out, (0, y0), (w-1, y0), (255, 255, 0), 1)

    if conf == 1:
        # anota contorno e centro
        cont_shift = cont + np.array([[[0, y0]]])
        cv2.drawContours(out, [cont_shift], -1, (0, 255, 0), 2)
        cv2.circle(out, (cx_full, cy_full), 6, (0, 0, 255), -1)
        cv2.line(out, (cx_img, h-1), (cx_full, cy_full), (255, 0, 0), 1)
        erro = cx_full - cx_img

    # deadband
    if abs(erro) <= DEAD_BAND:
        erro = 0

    return out, erro, conf

# ============================== CONTROLE (PI) ===============================
def lowpass(prev, new, dt, tau):
    """Filtro 1ª ordem (EMA) com constante de tempo tau (s)."""
    if dt <= 0:
        return new
    alpha = dt / (tau + dt)
    return prev + alpha * (new - prev)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# ============================ HARDWARE/PROTOCOLO ============================
def enviar_comando_motor_serial(arduino, v_esq, v_dir):
    # No FOLLOW, não permitimos ré; no LOST pode.
    comando = f"C {int(v_dir)} {int(v_esq)}\n"
    arduino.write(comando.encode('utf-8'))

# ================================== MAIN ====================================
def main():
    # --- ZMQ (vídeo + teclas) ---
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.bind('tcp://*:5555')

    req = context.socket(zmq.REQ)
    req.connect(f"tcp://{SERVER_IP}:5005")
    poller = zmq.Poller(); poller.register(req, zmq.POLLIN)
    awaiting = False; last_key = ""; prev_key = ""; last_req_ts = 0.0
    KEY_REQ_PERIOD = 0.05

    # --- Câmera ---
    cam = PiCamera(); cam.resolution = (IMG_WIDTH, IMG_HEIGHT); cam.framerate = 24
    raw = PiRGBArray(cam, size=(IMG_WIDTH, IMG_HEIGHT)); time.sleep(0.1)

    # --- Arduino ---
    ard = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=1); time.sleep(2)
    try:
        ard.write(b'A10\n')
        print("Arduino:", ard.readline().decode('utf-8').strip())
    except Exception:
        pass

    # Estados do seguidor
    mode = "AUTOMATICO"     # ou "MANUAL"
    state = "FOLLOW"        # FOLLOW | LOST
    lost_frames = 0
    found_streak = 0        # para histerese ao sair do LOST
    last_err = 0.0
    lost_dir = +1           # direção de giro corrente (+1 horário, -1 anti-horário)
    last_spin_flip = time.time()  # para alternar direção periodicamente

    # Controle PI
    e_filt = 0.0            # erro filtrado
    i_err = 0.0             # integrador
    last_t = time.time()

    v_esq = v_dir = 0

    print("Robo iniciado. Pressione 'm' para alternar modo.")

    try:
        for frame in cam.capture_continuous(raw, format="bgr", use_video_port=True):
            img = frame.array
            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            # ---------------- leitura de teclas (não bloqueante) ----------------
            if (not awaiting) and (now - last_req_ts >= KEY_REQ_PERIOD):
                try:
                    req.send_pyobj({"from": MY_ID, "cmd": "key_request"})
                    awaiting = True; last_req_ts = now
                except Exception:
                    pass
            socks = dict(poller.poll(0))
            if awaiting and req in socks and socks[req] == zmq.POLLIN:
                try:
                    rep = req.recv_pyobj(zmq.DONTWAIT)
                    last_key = rep.get("key", "")
                    awaiting = False
                except zmq.Again:
                    pass

            # toggle modo
            if last_key == 'm' and prev_key != 'm':
                mode = "MANUAL" if mode == "AUTOMATICO" else "AUTOMATICO"
                print("Modo:", mode)
                v_esq = v_dir = 0
                # zera integrador ao trocar modo
                i_err = 0.0
            prev_key = last_key

            # ----------------------------- CONTROLE -----------------------------
            conf = 0  # para HUD caso esteja em MANUAL
            if mode == "AUTOMATICO":
                img, erro, conf = process_image_for_control(img)

                if conf == 1:
                    # Histerese de saída do LOST
                    found_streak += 1
                    if found_streak >= FOUND_MIN_FRAMES:
                        state = "FOLLOW"
                        lost_frames = 0
                    # Atualiza erro filtrado (suaviza curvas)
                    e_filt = lowpass(e_filt, float(erro), dt, ERR_TAU)
                    last_err = e_filt

                    # Controle PI com anti-windup:
                    #  - integra apenas com conf==1
                    #  - aplica “leak” lento para esvaziar acúmulo residual
                    i_err += e_filt * dt
                    # anti-windup por clamp + leak
                    i_err = clamp(i_err, -I_MAX, I_MAX)
                    leak = max(0.0, 1.0 - dt / max(1e-6, I_LEAK_TAU))
                    i_err *= leak

                    steer = Kp * e_filt + Ki * i_err

                    # Programa de velocidade: reduz base com |erro|
                    base_scale = max(BASE_SCALE_MIN, 1.0 - abs(e_filt) / float(E_MAX_PIX))
                    base_speed = int(clamp(VELOCIDADE_BASE * base_scale, FORWARD_MIN, VELOCIDADE_MAX))

                    # Diferencial (sem permitir ré no FOLLOW)
                    v_esq = base_speed + steer
                    v_dir = base_speed - steer
                    v_esq = clamp(v_esq, 0, VELOCIDADE_MAX)
                    v_dir = clamp(v_dir, 0, VELOCIDADE_MAX)

                else:
                    # Sem detecção: entra no pipeline LOST
                    found_streak = 0
                    lost_frames += 1
                    if lost_frames >= LOST_MAX_FRAMES:
                        state = "LOST"

                    if state == "LOST":
                        # Alterna direção de giro periodicamente para não travar
                        if (now - last_spin_flip) >= LOST_SPIN_PERIOD:
                            # Se havia um último erro “grande”, use-o para iniciar
                            if abs(last_err) > DEAD_BAND:
                                lost_dir = 1 if last_err >= 0 else -1
                            else:
                                lost_dir *= -1
                            last_spin_flip = now
                        turn = SEARCH_SPEED * lost_dir
                        v_esq = clamp(LOST_FORWARD + turn, -VELOCIDADE_MAX, VELOCIDADE_MAX)
                        v_dir = clamp(LOST_FORWARD - turn, -VELOCIDADE_MAX, VELOCIDADE_MAX)
                        # zera integrador no LOST (evita “ficar doido” quando retornar)
                        i_err = 0.0
                    else:
                        # Janela de tolerância antes de LOST: avance bem devagar e reto
                        base_speed = int(clamp(VELOCIDADE_BASE * BASE_SCALE_MIN, FORWARD_MIN, VELOCIDADE_MAX))
                        v_esq = base_speed; v_dir = base_speed

            else:
                # --------------------------- MODO MANUAL -------------------------
                if last_key == 'w':   v_esq, v_dir = VELOCIDADE_BASE, VELOCIDADE_BASE
                elif last_key == 's': v_esq, v_dir = -VELOCIDADE_BASE, -VELOCIDADE_BASE
                elif last_key == 'a': v_esq, v_dir = -VELOCIDADE_BASE, VELOCIDADE_BASE
                elif last_key == 'd': v_esq, v_dir = VELOCIDADE_BASE, -VELOCIDADE_BASE
                elif last_key != '':  v_esq, v_dir = 0, 0
                # zera PI no manual para não herdar acúmulo
                i_err = 0.0
                e_filt = 0.0

            enviar_comando_motor_serial(ard, v_esq, v_dir)

            # ----------------------------- VISUALIZAÇÃO -------------------------
            disp = img.copy()
            h, w = disp.shape[:2]
            # HUD
            cv2.putText(disp, f"Modo: {mode}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(disp, f"State: {state}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 1)
            cv2.putText(disp, f"V_E:{int(v_esq)} V_D:{int(v_dir)}", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,0), 2)
            cv2.putText(disp, f"Conf: {conf}  LostFrames: {lost_frames}  FoundStreak: {found_streak}",
                        (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,255), 1)

            # publish frame
            _, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            pub.send(base64.b64encode(buf))
            raw.truncate(0)

    finally:
        print("Encerrando...")
        try:
            enviar_comando_motor_serial(ard, 0, 0)
            ard.write(b'a\n'); ard.close()
        except Exception:
            pass
        try:
            pub.close(); req.close(); context.term()
        except Exception:
            pass

if __name__ == "__main__":
    main()
