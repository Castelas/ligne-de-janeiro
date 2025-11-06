# robot2.py — seguidor de linha com giro 180 (NÃO-BLOQUEANTE)

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
SERVER_IP = "192.168.137.176"     # <--- COLOQUE o IP do servidor (control.py)
MY_ID     = "bot001"             # identificador do robô na rede

# --- VISÃO ---
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 180

# Hough (apenas para overlay/depuração visual — NÃO usado no controle)
HOUGHP_THRESHOLD    = 35
HOUGHP_MINLEN_FRAC  = 0.35
HOUGHP_MAXGAP       = 20
ROI_CROP_FRAC       = 0.20  # zera 20% do topo para reduzir ruído no overlay
RHO_MERGE           = 40
THETA_MERGE_DEG     = 6
ORTH_TOL_DEG        = 15
PAR_TOL_DEG         = 8

# --- CONTROLE (P puro; sem derivativo ainda) ---
VELOCIDADE_BASE = 150
VELOCIDADE_CURVA = 100

# --- AJUSTES DE ESTABILIDADE ---
Kp = 0.70 # Reduzido de 0.75 para um seguimento menos agressivo
VELOCIDADE_MAX = 255
MODO_AUTO   = "AUTOMATICO"
MODO_MANUAL = "MANUAL"

MODO_OBSTACLE_TURN = "TURNING_180"

# --- AJUSTES DO GIRO ---
# !! CALIBRAR ESTE VALOR !!
TURN_180_DURATION = 1.3 # Reduzido de 1.5s
TURN_180_SPEED = 90     # Reduzido de 100 (VELOCIDADE_CURVA)
# --- FIM AJUSTES ---

# Ajustes (detecção/recuperação)
E_MAX_PIX       = IMG_WIDTH // 2
V_MIN           = 0
SEARCH_SPEED    = 120
LOST_MAX_FRAMES = 5
DEAD_BAND       = 6
ROI_BOTTOM_FRAC = 0.55
MIN_AREA_FRAC   = 0.004
MAX_AREA_FRAC   = 0.25
ASPECT_MIN      = 2.0
LINE_POLARITY   = 'auto'
USE_ADAPTIVE    = False

# --- SERIAL ---
PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200

# ============================ AUXILIARES VISUAIS ============================
# (Funções _angle_diff, _deg, _dedup_points, build_binary_mask, 
#  detect_segments, segments_to_lines_rhotheta, line_intersection, 
#  detect_intersections... sem alterações)
def _angle_diff(a, b):
    d = abs((a - b) % np.pi)
    return min(d, np.pi - d)

def _deg(x): return np.deg2rad(x)

def _dedup_points(points, radius=25):
    if not points: return []
    used = [False]*len(points); out = []
    for i, p in enumerate(points):
        if used[i]: continue
        cluster = [p]; used[i] = True
        for j in range(i+1, len(points)):
            if (not used[j]) and (np.hypot(points[j][0]-p[0], points[j][1]-p[1]) < radius):
                used[j] = True; cluster.append(points[j])
        cx = int(np.mean([x for x,_ in cluster])); cy = int(np.mean([y for _,y in cluster]))
        out.append((cx, cy))
    return out

def build_binary_mask(image_bgr):
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    top = int(h * ROI_CROP_FRAC)
    mask[:top, :] = 0
    return mask

def detect_segments(mask):
    h, w = mask.shape[:2]
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    min_len = int(min(h, w) * HOUGHP_MINLEN_FRAC)
    seg = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGHP_THRESHOLD,
                          minLineLength=min_len, maxLineGap=HOUGHP_MAXGAP)
    if seg is None: return np.empty((0,4), dtype=int)
    return seg.reshape(-1, 4)

def segments_to_lines_rhotheta(segments):
    if len(segments) == 0: return []
    lines = []
    for x1, y1, x2, y2 in segments:
        ang_seg = np.arctan2((y2 - y1), (x2 - x1))
        theta = (ang_seg + np.pi/2) % np.pi
        rho = x1*np.cos(theta) + y1*np.sin(theta)
        lines.append((rho, theta))
    merged = []
    for rho, theta in lines:
        found = False
        for i, (r, t) in enumerate(merged):
            if abs(rho - r) < RHO_MERGE and _angle_diff(theta, t) < _deg(THETA_MERGE_DEG):
                merged[i] = ((rho + r)/2.0, (theta + t)/2.0); found = True; break
        if not found: merged.append((float(rho), float(theta)))
    return merged

def line_intersection(line1, line2):
    rho1, th1 = line1; rho2, th2 = line2
    a1, b1 = np.cos(th1), np.sin(th1)
    a2, b2 = np.cos(th2), np.sin(th2)
    det = a1*b2 - a2*b1
    if abs(det) < 1e-6: return None
    x = (b2*rho1 - b1*rho2)/det
    y = (a1*rho2 - a2*rho1)/det
    return (int(round(x)), int(round(y)))

def detect_intersections(mask):
    segments = detect_segments(mask)
    lines = segments_to_lines_rhotheta(segments)
    if not lines: return [], []
    vertical   = [l for l in lines if _angle_diff(l[1], 0.0) < _deg(15)]
    horizontal = [l for l in lines if _angle_diff(l[1], np.pi/2) < _deg(15)]
    H, W = mask.shape[:2]
    pts = []
    for lv in vertical:
        for lh in horizontal:
            p = line_intersection(lv, lh)
            if p is None: continue
            x, y = p
            if 0 <= x < W and 0 <= y < H: pts.append((x, y))
    pts = _dedup_points(pts, radius=25)
    return pts, (vertical + horizontal)

# ====================== DETECÇÃO PARA CONTROLE (ROI/CONFIANÇA) =============
# (Função processar_imagem... sem alterações)
def processar_imagem(imagem):
    h, w = imagem.shape[:2]
    cx_img = w // 2
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if USE_ADAPTIVE:
        th_white = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 31, -5)
        th_black = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 31, -5)
    else:
        _, th_white = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        _, th_black = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)

    def find_valid_contour(th):
        y0 = int(h * ROI_BOTTOM_FRAC)
        roi = th[y0:h, :]
        eroded = cv2.erode(roi, None, iterations=1)
        dilated = cv2.dilate(eroded, None, iterations=1)
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None, None, 0
        roi_area = w * (h - y0)
        best = None; best_len = -1.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < roi_area * MIN_AREA_FRAC:  continue
            if area > roi_area * MAX_AREA_FRAC:  continue
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            if rw < 1 or rh < 1: continue
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            if aspect < ASPECT_MIN: continue
            length = max(rw, rh)
            if length > best_len:
                best_len = length; best = c
        if best is None: return None, None, None, 0
        M = cv2.moments(best)
        if M["m00"] <= 1e-6: return None, None, None, 0
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        cx_full, cy_full = cx, cy + y0
        return best, cx_full, cy_full, 1

    if LINE_POLARITY == 'white':
        c, cx_full, cy_full, conf = find_valid_contour(th_white)
    elif LINE_POLARITY == 'black':
        c, cx_full, cy_full, conf = find_valid_contour(th_black)
    else:
        c, cx_full, cy_full, conf = find_valid_contour(th_white)
        if conf == 0:
            c, cx_full, cy_full, conf = find_valid_contour(th_black)
    erro = 0
    if conf == 1:
        y0 = int(h * ROI_BOTTOM_FRAC)
        c_shifted = c + np.array([[[0, y0]]])
        cv2.drawContours(imagem, [c_shifted], -1, (0, 255, 0), 2)
        cv2.circle(imagem, (cx_full, cy_full), 7, (0, 0, 255), -1)
        cv2.line(imagem, (cx_img, h-1), (cx_full, cy_full), (255, 0, 0), 1)
        erro = cx_full - cx_img
    if abs(erro) <= DEAD_BAND:
        erro = 0
    return imagem, erro, conf

# ============================ CONTROLE (P) ==================================
# (Função calcular_velocidades_auto... sem alterações)
def calcular_velocidades_auto(erro, base_speed):
    correcao = Kp * float(erro)
    v_esq = base_speed + correcao
    v_dir = base_speed - correcao
    v_esq = int(np.clip(v_esq, 15, VELOCIDADE_MAX))
    v_dir = int(np.clip(v_dir, 15, VELOCIDADE_MAX))
    return v_esq, v_dir

# --- MUDANÇA: Lógica Não-Bloqueante ---
# Esta variável global guardará o último estado reportado pelo Arduino
# Acessada por 'enviar_comando_motor_serial' e 'main'
resposta_arduino_global = "OK"

def enviar_comando_motor_serial(arduino, v_esq, v_dir):
    global resposta_arduino_global
    
    # 1. Envia o comando do motor (não bloqueante)
    comando = f"C {v_dir} {v_esq}\n"
    arduino.write(comando.encode('utf-8'))
    
    # 2. Verifica se há dados na fila para ler (não bloqueante)
    try:
        if arduino.in_waiting > 0:
            # 3. Se houver, lê a resposta (ex: "OK" ou "OB")
            resposta = arduino.readline().decode('utf-8').strip()
            
            # 4. Atualiza o estado global apenas se for uma resposta válida
            if resposta in ["OK", "OB"]:
                resposta_arduino_global = resposta
            
            # Opcional: Esvazia o buffer de entrada caso algo
            #           estranho tenha se acumulado
            arduino.reset_input_buffer()
            
    except Exception as e:
        # Ignora erros de serial (ex: desconexão)
        pass 
# --- FIM MUDANÇA ---


# ================================= MAIN =====================================
def main():
    global resposta_arduino_global # Permite que a função de envio altere esta variável
    
    # --- ZMQ ---
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind('tcp://*:5555')
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(f"tcp://{SERVER_IP}:5005")
    poller = zmq.Poller(); poller.register(req_socket, zmq.POLLIN)
    awaiting_reply = False; last_key = ""; prev_key = ""; last_req_ts = 0.0
    KEY_REQ_PERIOD = 0.05

    # --- Câmera ---
    camera = PiCamera()
    camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
    camera.framerate = 24
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    time.sleep(0.1)

    # --- Arduino ---
    # --- MUDANÇA: Timeout de 0.1s. Não precisamos mais de 1s. ---
    arduino = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=0.1); time.sleep(2)
    try:
        # --- MUDANÇA: CORREÇÃO DE BUG (Adiciona \n) ---
        arduino.write(b'A10') 
        print(f"Arduino: {arduino.readline().decode('utf-8').strip()}")
        
        # Ativa a task4 (detecção de obstáculo)
        arduino.write(b'I1') 
        print(f"Protecao: {arduino.readline().decode('utf-8').strip()}")
        # --- FIM MUDANÇA ---
        
    except Exception as e:
        print(f"Erro inicial Arduino: {e}")
        pass

    # Estado do seguidor
    last_err = 0.0
    lost_frames = 0
    state = 'FOLLOW'
    # resposta_arduino = "OK" # <--- Não é mais necessário, usamos a global
    
    # Timer para o giro
    turn_start_time = 0.0
    
    current_mode = MODO_AUTO
    v_esq, v_dir = 0, 0
    print("Robo iniciado. Pressione 'm' para trocar de modo.")

    try:
        for frame in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            image = frame.array

            # ---------- Teclas (não bloqueante) ----------
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
            
            # Lógica de toggle manual (tem prioridade)
            toggled = (key == 'm' and prev_key != 'm'); prev_key = key
            if toggled:
                current_mode = MODO_MANUAL if current_mode == MODO_AUTO else MODO_AUTO
                print(f"Modo: {current_mode}")
                v_esq, v_dir = 0, 0
                state = 'FOLLOW'      # Reseta estado
                turn_start_time = 0.0 # Reseta timer de giro
            
            conf = 0  # default p/ HUD

            # ----------------- CONTROLE -----------------
            if current_mode == MODO_AUTO:
                
                # --- MUDANÇA: Lógica de estado agora lê a variável GLOBAL ---
                
                # Se NÃO estamos girando, verificamos se há um novo obstáculo
                if state != MODO_OBSTACLE_TURN and resposta_arduino_global == "OB":
                    print("OBSTACULO DETECTADO! Iniciando giro 180...")
                    state = MODO_OBSTACLE_TURN
                    turn_start_time = time.time() # Inicia o timer
                    last_err = 0 # Zera último erro para o próximo LOST
                    
                
                if state == MODO_OBSTACLE_TURN:
                    # --- ESTADO 1: GIRANDO 180 ---
                    now = time.time()
                    if (now - turn_start_time) < TURN_180_DURATION:
                        # Ainda girando...
                        v_esq = TURN_180_SPEED
                        v_dir = -TURN_180_SPEED
                        erro, conf = 0, 0 # Zera para HUD
                    else:
                        # Giro completo!
                        print("Giro completo. Procurando linha...")
                        v_esq, v_dir = 0, 0
                        state = 'LOST' # Entra em modo LOST para achar a linha
                        turn_start_time = 0.0 # Reseta timer
                        lost_frames = 0 # Força re-busca
                
                else:
                    # --- ESTADOS 2 (FOLLOW) e 3 (LOST) ---
                    image, erro, conf = processar_imagem(image)

                    if conf == 1:
                        # Detecção válida: FOLLOW
                        state = 'FOLLOW'
                        lost_frames = 0
                        last_err = erro
                        
                        speed_scale = max(0.35, 1.0 - abs(erro) / float(E_MAX_PIX))
                        base_speed = int(np.clip(VELOCIDADE_BASE * speed_scale, V_MIN, VELOCIDADE_MAX))
                        v_esq, v_dir = calcular_velocidades_auto(erro, base_speed)
                    else:
                        # Sem detecção:
                        lost_frames += 1
                        if lost_frames >= LOST_MAX_FRAMES:
                            state = 'LOST'

                        if state == 'LOST':
                            # Busca ativa
                            turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                            v_esq, v_dir = int(turn), int(-turn)
                        else:
                            # Tolerância
                            base_speed = int(np.clip(VELOCIDADE_BASE * 0.35, V_MIN, VELOCIDADE_MAX))
                            v_esq, v_dir = calcular_velocidades_auto(0, base_speed)
            
            else:
                # Modo manual
                if key == 'w':   v_esq, v_dir = VELOCIDADE_BASE, VELOCIDADE_BASE
                elif key == 's': v_esq, v_dir = -VELOCIDADE_BASE, -VELOCIDADE_BASE
                elif key == 'a': v_esq, v_dir = -VELOCIDADE_CURVA, VELOCIDADE_CURVA
                elif key == 'd': v_esq, v_dir = VELOCIDADE_CURVA, -VELOCIDADE_CURVA
                elif key != '':  v_esq, v_dir = 0, 0

            # --- MUDANÇA: A chamada de envio agora é NÃO-BLOQUEANTE ---
            # Ela envia o comando E atualiza 'resposta_arduino_global'
            # se algo for lido do buffer.
            enviar_comando_motor_serial(arduino, v_esq, v_dir)
            # --- FIM MUDANÇA ---


            # ---------------- VISUALIZAÇÃO ----------------
            display_frame = image.copy()
            mask = build_binary_mask(display_frame)
            intersections, detected_lines = detect_intersections(mask)

            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

            for rho, theta in detected_lines:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1 = int(x0 + 1000 * (-b));  y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b));  y2 = int(y0 - 1000 * (a))
                cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for idx, (x, y) in enumerate(intersections, 1):
                cv2.circle(display_frame, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(display_frame, (x, y), 12, (255, 255, 255), 2)
                cv2.putText(display_frame, f"{idx}", (x + 15, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Modo: {current_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"V_E:{v_esq} V_D:{v_dir}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            
            hud_color = (0, 200, 255) 
            if state == MODO_OBSTACLE_TURN:
                hud_color = (0, 0, 255)
            elif state == 'LOST':
                hud_color = (0, 165, 255)
            elif state == 'FOLLOW':
                hud_color = (0, 255, 0) 
                
            cv2.putText(display_frame, f"State: {state}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
            
            cv2.putText(display_frame, f"Lines: {len(detected_lines)}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(display_frame, f"Intersections: {len(intersections)}", (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(display_frame, f"Conf: {conf}  LostFrames: {lost_frames}", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)

            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            pub_socket.send(base64.b64encode(buffer))
            raw.truncate(0)

    finally:
        print("Encerrando...")
        try:
            enviar_comando_motor_serial(arduino, 0, 0)
            # --- MUDANÇA: CORREÇÃO DE BUG (Adiciona \n) ---
            arduino.write(b'a'); arduino.close() 
        except Exception:
            pass
        try:
            pub_socket.close(); req_socket.close()
        except Exception:
            pass
        try:
            context.term()
        except Exception:
            pass

if __name__ == "__main__":
    main()