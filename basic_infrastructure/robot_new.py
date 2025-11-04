# robot.py (detecção reescrita com HoughLinesP + agrupamento por orientação)
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import zmq
import base64
import time
import numpy as np
import serial
from itertools import combinations

# ###########################################################################
# ##############           PARÂMETROS DE CALIBRAGEM           ###############
# ###########################################################################
# --- PARÂMETROS DE REDE ---
SERVER_IP = "192.168.137.95"  # <--- MUDE PARA O IP DO SEU SERVIDOR
MY_ID = 'bot001'

# --- PARÂMETROS DE VISÃO ---
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 180

# HoughLinesP (ajuste se necessário)
HOUGHP_THRESHOLD = 35          # votos mínimos
HOUGHP_MINLEN_FRAC = 0.35      # fração do menor lado da imagem (minLineLength)
HOUGHP_MAXGAP = 20             # distância para ligar segmentos

# Merge e ortogonalidade
RHO_MERGE = 40                 # px
THETA_MERGE_DEG = 6            # °
ORTH_TOL_DEG = 15              # °
PAR_TOL_DEG = 8                # ° (para considerar paralelas)

# ROI (zera o topo para reduzir ruído no overlay de Hough)
ROI_CROP_FRAC = 0.20           # zera 20% superior (mais conservador que 33%)

# --- PARÂMETROS DE CONTROLE ---
VELOCIDADE_BASE = 150
VELOCIDADE_CURVA = 100
Kp = 0.8
VELOCIDADE_MAX = 255
MODO_AUTO = "AUTOMATICO"
MODO_MANUAL = "MANUAL"

# Ajustes de detecção/controle (sem alterar a lei de controle)
E_MAX_PIX = IMG_WIDTH // 2   # erro máximo considerado (pixels)
V_MIN = 0                    # velocidade mínima permitida no modo AUTO
SEARCH_SPEED = 120           # velocidade para procura girando no lugar
LOST_MAX_FRAMES = 5          # quantos frames sem linha até entrar em LOST
DEAD_BAND = 6                # pixels próximos do centro considerados zero
ROI_BOTTOM_FRAC = 0.55       # inicia ROI no 55% da altura para o controle (faixa perto do robô)
MIN_AREA_FRAC = 0.004        # fração mínima de área do contorno válida
USE_ADAPTIVE = False         # manter threshold fixo por padrão

# --- PARÂMETROS DE COMUNICAÇÃO SERIAL ---
PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200
# ###########################################################################

# ============================ UTIL/ÂNGULOS =================================
def _angle_diff(a, b):
    d = abs((a - b) % np.pi)
    return min(d, np.pi - d)

def _deg(x): return np.deg2rad(x)

def _is_parallel(a, b, tol=_deg(PAR_TOL_DEG)):
    return _angle_diff(a, b) < tol

def _is_orthogonal(a, b, tol=_deg(ORTH_TOL_DEG)):
    return abs(_angle_diff(a, b) - np.pi/2) < tol

def _dedup_points(points, radius=25):
    if not points:
        return []
    used = [False]*len(points)
    out = []
    for i, p in enumerate(points):
        if used[i]: 
            continue
        cluster = [p]; used[i] = True
        for j in range(i+1, len(points)):
            if (not used[j]) and (np.hypot(points[j][0]-p[0], points[j][1]-p[1]) < radius):
                used[j] = True; cluster.append(points[j])
        cx = int(np.mean([x for x,_ in cluster]))
        cy = int(np.mean([y for _,y in cluster]))
        out.append((cx, cy))
    return out

# ========================= MÁSCARA/SEGMENTOS ===============================
def build_binary_mask(image_bgr):
    """Cinza -> blur -> threshold -> morfologia + ROI (para overlay/linhas)."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    # ROI: zera apenas 20% superior (mantém mais da faixa vertical)
    top = int(h * ROI_CROP_FRAC)
    mask[:top, :] = 0
    return mask

def detect_segments(mask):
    """Hough probabilístico (segmentos)."""
    h, w = mask.shape[:2]
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    min_len = int(min(h, w) * HOUGHP_MINLEN_FRAC)
    seg = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGHP_THRESHOLD,
                          minLineLength=min_len, maxLineGap=HOUGHP_MAXGAP)
    if seg is None:
        return np.empty((0,4), dtype=int)
    return seg.reshape(-1, 4)

def segments_to_lines_rhotheta(segments):
    """
    Converte segmentos (x1,y1,x2,y2) para (rho,theta) de uma linha infinita
    e faz merge de linhas próximas.
    """
    if len(segments) == 0:
        return []

    lines = []
    for x1, y1, x2, y2 in segments:
        # Ângulo do segmento em relação ao eixo x
        ang_seg = np.arctan2((y2 - y1), (x2 - x1))  # [-pi, pi]
        # Para forma normal: theta = ang_seg + 90°, mod pi
        theta = (ang_seg + np.pi/2) % np.pi
        # Rho calculado a partir de qualquer ponto do segmento
        rho = x1*np.cos(theta) + y1*np.sin(theta)
        lines.append((rho, theta))

    # Merge
    merged = []
    for rho, theta in lines:
        found = False
        for i, (r, t) in enumerate(merged):
            if abs(rho - r) < RHO_MERGE and _angle_diff(theta, t) < _deg(THETA_MERGE_DEG):
                merged[i] = ((rho + r)/2.0, (theta + t)/2.0)
                found = True
                break
        if not found:
            merged.append((float(rho), float(theta)))
    return merged

# ============================ INTERSEÇÕES ==================================
def line_intersection(line1, line2):
    rho1, th1 = line1
    rho2, th2 = line2
    if _is_parallel(th1, th2):  # evita paralelas
        return None
    a1, b1 = np.cos(th1), np.sin(th1)
    a2, b2 = np.cos(th2), np.sin(th2)
    det = a1*b2 - a2*b1
    if abs(det) < 1e-6:
        return None
    x = (b2*rho1 - b1*rho2)/det
    y = (a1*rho2 - a2*rho1)/det
    return (int(round(x)), int(round(y)))

def detect_intersections(mask):
    """
    1) Detecta segmentos (HoughLinesP)
    2) Converte para (rho,theta) e agrupa
    3) Separa por orientação (vertical: theta≈0; horizontal: theta≈pi/2)
    4) Interseções apenas entre vertical × horizontal
    """
    segments = detect_segments(mask)
    lines = segments_to_lines_rhotheta(segments)
    if not lines:
        return [], []

    # Classificação por orientação da linha: theta≈0 => vertical, theta≈pi/2 => horizontal
    vertical = [l for l in lines if _angle_diff(l[1], 0.0) < _deg(15)]
    horizontal = [l for l in lines if _angle_diff(l[1], np.pi/2) < _deg(15)]

    H, W = mask.shape[:2]
    pts = []
    for lv in vertical:
        for lh in horizontal:
            p = line_intersection(lv, lh)
            if p is None: 
                continue
            x, y = p
            if 0 <= x < W and 0 <= y < H:
                pts.append((x, y))

    pts = _dedup_points(pts, radius=25)
    # Retorna todas as linhas (vert+horz) para desenhar/contar
    return pts, (vertical + horizontal)

# ====================== PROCESSAMENTO E CONTROLE ============================
def processar_imagem(imagem):
    """
    Calcula o erro lateral da faixa usando uma ROI inferior da imagem (mais estável).
    Retorna (imagem_annot, erro_pixels, conf)
      - erro_pixels: deslocamento do centroide da faixa em relação ao centro da imagem
      - conf: 1 se um contorno válido foi encontrado, 0 caso contrário
    """
    h, w = imagem.shape[:2]
    cx_img = w // 2

    # Cinza + blur
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold (fixo por padrão; há opção adaptativa se desejar)
    if USE_ADAPTIVE:
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, -5)
    else:
        _, th = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # ROI inferior (descarta o topo para focar perto do robô)
    y0 = int(h * ROI_BOTTOM_FRAC)
    roi = th[y0:h, :]

    # Morfologia
    eroded = cv2.erode(roi, None, iterations=1)
    dilated = cv2.dilate(eroded, None, iterations=1)

    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    conf = 0
    erro = 0
    if len(contours) > 0:
        # Escolhe o maior contorno, mas checa área mínima
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area >= (w * (h - y0)) * MIN_AREA_FRAC:
            M = cv2.moments(c)
            if M["m00"] > 1e-6:
                cx = int(M["m10"] / M["m00"])  # coordenada dentro da ROI
                cy = int(M["m01"] / M["m00"])  # coordenada dentro da ROI
                # Converte para coordenadas da imagem completa
                cx_full = cx
                cy_full = cy + y0

                # Anotações
                c_shifted = c + np.array([[[0, y0]]])
                cv2.drawContours(imagem, [c_shifted], -1, (0, 255, 0), 2)
                cv2.circle(imagem, (cx_full, cy_full), 7, (0, 0, 255), -1)
                cv2.line(imagem, (cx_img, h-1), (cx_full, cy_full), (255, 0, 0), 1)

                erro = cx_full - cx_img
                conf = 1

    # Aplica deadband para reduzir jitter
    if abs(erro) <= DEAD_BAND:
        erro = 0

    return imagem, erro, conf

def calcular_velocidades_auto(erro, base_speed):
    # Lei P: mesma lógica, apenas com base variável e saturação simétrica (permite ré)
    correcao = Kp * float(erro)
    v_esq = base_speed + correcao
    v_dir = base_speed - correcao
    v_esq = int(np.clip(v_esq, -VELOCIDADE_MAX, VELOCIDADE_MAX))
    v_dir = int(np.clip(v_dir, -VELOCIDADE_MAX, VELOCIDADE_MAX))
    return v_esq, v_dir

def enviar_comando_motor_serial(arduino, v_esq, v_dir):
    # Envia velocidades com sinal; negativos significam ré (Arduino deve aceitar número com sinal)
    comando = f"C {v_dir} {v_esq}\n"
    arduino.write(comando.encode('utf-8'))

# =============================== MAIN ======================================
def main():
    # --- ZMQ ---
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB); pub_socket.bind('tcp://*:5555')

    req_socket = context.socket(zmq.REQ); req_socket.connect(f"tcp://{SERVER_IP}:5005")
    poller = zmq.Poller(); poller.register(req_socket, zmq.POLLIN)
    awaiting_reply = False; last_key = ""; prev_key = ""; last_req_ts = 0.0
    KEY_REQ_PERIOD = 0.05

    # --- Câmera ---
    camera = PiCamera(); camera.resolution = (IMG_WIDTH, IMG_HEIGHT); camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT)); time.sleep(0.1)

    # --- Arduino ---
    arduino = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=1); time.sleep(2)
    arduino.write(b'A10\n')
    print(f"Arduino respondeu: {arduino.readline().decode('utf-8').strip()}")

    # Estado do seguidor de linha (sem derivativo ainda)
    last_err = 0.0
    lost_frames = 0
    state = 'FOLLOW'  # FOLLOW | LOST

    current_mode = MODO_AUTO
    v_esq, v_dir = 0, 0
    print("Robo iniciado. Pressione 'm' para trocar de modo.")

    try:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array

            # ---------- POLLING DE TECLAS (NAO BLOQUEANTE) ----------
            now = time.time()
            if not awaiting_reply and (now - last_req_ts) >= KEY_REQ_PERIOD:
                try:
                    req_socket.send_pyobj({"from": MY_ID, "cmd": "key_request"})
                    awaiting_reply = True; last_req_ts = now
                except Exception:
                    pass
            if awaiting_reply:
                socks = dict(poller.poll(0))
                if req_socket in socks and socks[req_socket] == zmq.POLLIN:
                    try:
                        reply = req_socket.recv_pyobj(zmq.DONTWAIT)
                        last_key = reply.get("key", "")
                        awaiting_reply = False
                    except zmq.Again:
                        pass

            key = last_key
            toggled = (key == 'm' and prev_key != 'm'); prev_key = key
            if toggled:
                current_mode = MODO_MANUAL if current_mode == MODO_AUTO else MODO_AUTO
                print(f"Modo alterado para: {current_mode}")
                v_esq, v_dir = 0, 0

            # ----------------- CONTROLE -----------------
            if current_mode == MODO_AUTO:
                # Processa imagem para erro lateral (com confiança)
                image, erro, conf = processar_imagem(image)

                if conf == 0:
                    lost_frames += 1
                    if lost_frames >= LOST_MAX_FRAMES:
                        state = 'LOST'
                else:
                    state = 'FOLLOW'

                if state == 'FOLLOW':
                    # Reset contador de perda e guarda último erro (para decidir giro)
                    lost_frames = 0
                    last_err = erro

                    # Programa de velocidade: diminui quando erro cresce
                    speed_scale = max(0.35, 1.0 - abs(erro) / float(E_MAX_PIX))
                    base_speed = int(np.clip(VELOCIDADE_BASE * speed_scale, V_MIN, VELOCIDADE_MAX))

                    v_esq, v_dir = calcular_velocidades_auto(erro, base_speed)
                else:  # LOST
                    # Procura a faixa: gira no lugar para o lado do último erro conhecido
                    turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                    v_esq, v_dir = int(turn), int(-turn)
            else:
                if key == 'w':   v_esq, v_dir = VELOCIDADE_BASE, VELOCIDADE_BASE
                elif key == 's': v_esq, v_dir = -VELOCIDADE_BASE, -VELOCIDADE_BASE
                elif key == 'a': v_esq, v_dir = -VELOCIDADE_CURVA, VELOCIDADE_CURVA
                elif key == 'd': v_esq, v_dir = VELOCIDADE_CURVA, -VELOCIDADE_CURVA
                elif key != '':  v_esq, v_dir = 0, 0

            enviar_comando_motor_serial(arduino, v_esq, v_dir)

            # --------------- VISUALIZAÇÃO ---------------
            display_frame = image.copy()
            mask = build_binary_mask(display_frame)
            intersections, detected_lines = detect_intersections(mask)

            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

            # desenha linhas detectadas (verde)
            for rho, theta in detected_lines:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1 = int(x0 + 1000 * (-b));  y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b));  y2 = int(y0 - 1000 * (a))
                cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # interseções (vermelho)
            for idx, (x, y) in enumerate(intersections, 1):
                cv2.circle(display_frame, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(display_frame, (x, y), 12, (255, 255, 255), 2)
                cv2.putText(display_frame, f"{idx}", (x + 15, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Modo: {current_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"V_E:{v_esq} V_D:{v_dir}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(display_frame, f"State: {state}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(display_frame, f"Lines: {len(detected_lines)}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(display_frame, f"Intersections: {len(intersections)}", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            pub_socket.send(base64.b64encode(buffer))
            rawCapture.truncate(0)

    finally:
        print("Encerrando...")
        try:
            enviar_comando_motor_serial(arduino, 0, 0)
            arduino.write(b'a\n')
            arduino.close()
        except Exception:
            pass
        pub_socket.close(); req_socket.close(); context.term()

if __name__ == "__main__":
    main()
