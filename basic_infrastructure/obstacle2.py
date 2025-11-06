# robot2.py — seguidor de linha com ROI, confiança e estado FOLLOW/LOST (sem derivativo)

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import zmq
import base64
import time
import numpy as np
import serial
import threading
import queue

# ============================= PARÂMETROS GERAIS =============================
# --- REDE ---
SERVER_IP = "192.168.137.22"     # <--- COLOQUE o IP do servidor (control.py)
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
Kp = 0.75
VELOCIDADE_MAX = 255
MODO_AUTO   = "AUTOMATICO"
MODO_MANUAL = "MANUAL"

# Ajustes (detecção/recuperação)
E_MAX_PIX       = IMG_WIDTH // 2        # erro máximo usado para escalonar velocidade
V_MIN           = 0                     # velocidade mínima admitida no AUTO
SEARCH_SPEED    = 120                   # velocidade para girar no lugar em LOST
LOST_MAX_FRAMES = 5                     # frames sem confiança até entrar em LOST
DEAD_BAND       = 6                     # |erro| <= DEAD_BAND => erro = 0
ROI_BOTTOM_FRAC = 0.55                  # início da ROI inferior (55% da altura)
MIN_AREA_FRAC   = 0.004                 # área mínima do contorno na ROI (fração)
MAX_AREA_FRAC   = 0.25                  # área máxima aceitável (descarta “piso inteiro”)
ASPECT_MIN      = 2.0                   # formato “faixa”: comprimento/largura mínimo
LINE_POLARITY   = 'auto'                # 'white', 'black' ou 'auto'
USE_ADAPTIVE    = False                 # threshold adaptativo desligado por padrão

# --- SERIAL ---
PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200

# --- ESTADOS DO ROBÔ ---
ESTADO_SEGUINDO  = "FOLLOW"
ESTADO_PERDIDO   = "LOST"
ESTADO_OBSTACULO = "OBSTACLE"

# --- DETECÇÃO DE OBSTÁCULOS ---
DISTANCIA_PARADA_CM = 15
OBSTACLE_CHECK_INTERVAL = 0.1 # Podemos verificar mais rápido agora


# ============================ WORKER SERIAL (NOVA SEÇÃO) ============================

def serial_worker(arduino, command_queue, robot_state, stop_event):
    """
    Esta função roda em uma thread separada.
    Seu único trabalho é gerenciar a comunicação com o Arduino.
    """
    last_sensor_request_time = 0
    SENSOR_REQUEST_INTERVAL = 0.1 # Pede a distância a cada 100ms

    while not stop_event.is_set():
        # 1. Enviar comandos da fila
        try:
            command = command_queue.get_nowait()
            arduino.write(command)
        except queue.Empty:
            pass # Fila vazia, sem problemas

        # 2. Pedir dados do sensor periodicamente
        now = time.time()
        if now - last_sensor_request_time > SENSOR_REQUEST_INTERVAL:
            try:
                arduino.write(b'S\n')
                # Esta é a única chamada bloqueante, mas está segura na sua própria thread
                response = arduino.readline().decode('utf-8').strip()
                if response:
                    dist = int(response)
                    # Atualiza o estado compartilhado
                    robot_state['distance'] = dist
                last_sensor_request_time = now
            except (ValueError, serial.SerialException) as e:
                print(f"Erro na thread serial: {e}")
        
        time.sleep(0.01) # Pequena pausa para não sobrecarregar o processador

# ============================ AUXILIARES VISUAIS ============================
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
    """Máscara binária para overlay de linhas/interseções (não usada no controle)."""
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
    # separa quase-verticais (theta~0) e quase-horizontais (theta~pi/2)
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
def processar_imagem(imagem):
    """
    Retorna (frame_annotado, erro_pixels, conf)
      - erro_pixels = cx_da_faixa - centro_da_imagem
      - conf: 1 se um contorno válido foi encontrado, 0 caso contrário
    """
    h, w = imagem.shape[:2]
    cx_img = w // 2

    # Cinza + blur
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholds (duas polaridades)
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
            if area < roi_area * MIN_AREA_FRAC:  # muito pequeno
                continue
            if area > roi_area * MAX_AREA_FRAC:  # muito grande (piso)
                continue

            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            if rw < 1 or rh < 1: 
                continue
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            if aspect < ASPECT_MIN: 
                continue

            length = max(rw, rh)
            if length > best_len:
                best_len = length; best = c

        if best is None: 
            return None, None, None, 0

        M = cv2.moments(best)
        if M["m00"] <= 1e-6: 
            return None, None, None, 0
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        cx_full, cy_full = cx, cy + y0
        return best, cx_full, cy_full, 1

    # Seleção por polaridade
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

    # Deadband
    if abs(erro) <= DEAD_BAND:
        erro = 0

    return imagem, erro, conf

# ============================ CONTROLE (P) ==================================
def calcular_velocidades_auto(erro, base_speed):
    """Lei P com base variável e saturação simétrica (permite ré)."""
    correcao = Kp * float(erro)
    v_esq = base_speed + correcao
    v_dir = base_speed - correcao
    v_esq = int(np.clip(v_esq, 15, VELOCIDADE_MAX))
    v_dir = int(np.clip(v_dir, 15, VELOCIDADE_MAX))
    return v_esq, v_dir

def enviar_comando_motor(command_queue, v_esq, v_dir):
    """Coloca um comando de motor na fila para ser enviado pela thread serial."""
    comando = f"C {v_dir} {v_esq}\n".encode('utf-8')
    command_queue.put(comando)

def ativar_camada_seguranca(command_queue):
    """Ativa a tarefa de segurança no Arduino."""
    print("Ativando camada de segurança no Arduino (I1)")
    command_queue.put(b'I1\n')

def desativar_camada_seguranca(command_queue):
    """Desativa a tarefa de segurança no Arduino."""
    print("Desativando camada de segurança no Arduino (I0)")
    command_queue.put(b'I0\n')

# ================================= MAIN =====================================
def main():
    # --- OBJETOS COMPARTILHADOS ENTRE THREADS ---
    command_queue = queue.Queue()
    robot_state = {'distance': 999}
    stop_event = threading.Event()

    # --- ZMQ ---
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind('tcp://*:5555')                     # publica vídeo

    req_socket = context.socket(zmq.REQ)                # consulta teclas
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

    # --- Arduino e Thread Serial ---
    try:
        # Timeout pode ser maior agora, pois não bloqueia o loop principal
        arduino = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=0.5)
        time.sleep(2)
        command_queue.put(b'A10\n') # Conecta com feedback
        print(f"Arduino: {arduino.readline().decode('utf-g').strip()}") # Leitura inicial OK
    except Exception as e:
        print(f"Falha na inicialização do Arduino: {e}")
        return

    # Inicia a thread de comunicação
    comm_thread = threading.Thread(target=serial_worker, args=(arduino, command_queue, robot_state, stop_event))
    comm_thread.daemon = True
    comm_thread.start()
    
    ativar_camada_seguranca(command_queue)

    # Estado do seguidor (sem derivativo)
    last_err = 0.0
    lost_frames = 0
    state = 'FOLLOW'  # FOLLOW | LOST

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
            toggled = (key == 'm' and prev_key != 'm'); prev_key = key
            if toggled:
                current_mode = MODO_MANUAL if current_mode == MODO_AUTO else MODO_AUTO
                print(f"Modo: {current_mode}")
                v_esq, v_dir = 0, 0

            conf = 0  # default p/ HUD caso esteja no modo MANUAL

            # ----------------- CONTROLE -----------------
            if current_mode == MODO_AUTO:
                # 1. Lê a distância do estado compartilhado (sem bloqueio!)
                distancia_obstaculo = robot_state['distance']
                
                 # 2. Lógica da máquina de estados
                if distancia_obstaculo < DISTANCIA_PARADA_CM:
                    if state != ESTADO_OBSTACULO:
                        print(f"OBSTÁCULO DETECTADO a {distancia_obstaculo} cm! Parando.")
                        state = ESTADO_OBSTACULO
                elif state == ESTADO_OBSTACULO and distancia_obstaculo >= DISTANCIA_PARADA_CM + 5:
                    print("Obstáculo removido. Retomando operação...")
                    ativar_camada_seguranca(command_queue) # Reativa para resetar a flag 'obst'
                    state = ESTADO_PERDIDO

                if state == ESTADO_OBSTACULO:
                    v_esq, v_dir = 0, 0
                
                else:
                    image, erro, conf = processar_imagem(image)
                    if state == ESTADO_SEGUINDO:
                        if conf == 1:
                            lost_frames = 0; last_err = erro
                            speed_scale = max(0.35, 1.0 - abs(erro) / float(E_MAX_PIX))
                            base_speed = int(np.clip(VELOCIDADE_BASE * speed_scale, V_MIN, VELOCIDADE_MAX))
                            v_esq, v_dir = calcular_velocidades_auto(erro, base_speed)
                        else:
                            lost_frames += 1
                            if lost_frames >= LOST_MAX_FRAMES:
                                state = ESTADO_PERDIDO
                            v_esq, v_dir = calcular_velocidades_auto(0, int(VELOCIDADE_BASE * 0.35))
                    elif state == ESTADO_PERDIDO:
                        if conf == 1:
                            state = ESTADO_SEGUINDO
                            lost_frames = 0
                        else:
                            turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                            v_esq, v_dir = int(turn), int(-turn)
            else: # MODO_MANUAL
                if key == 'w':   v_esq, v_dir = VELOCIDADE_BASE, VELOCIDADE_BASE
                elif key == 's': v_esq, v_dir = -VELOCIDADE_BASE, -VELOCIDADE_BASE
                elif key == 'a': v_esq, v_dir = -VELOCIDADE_CURVA, VELOCIDADE_CURVA
                elif key == 'd': v_esq, v_dir = VELOCIDADE_CURVA, -VELOCIDADE_CURVA
                elif key != '':  v_esq, v_dir = 0, 0

            enviar_comando_motor(command_queue, v_esq, v_dir)

            # ---------------- VISUALIZAÇÃO ----------------
            display_frame = image.copy()
            mask = build_binary_mask(display_frame)
            intersections, detected_lines = detect_intersections(mask)

            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

            # desenha linhas (verde)
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
            cv2.putText(display_frame, f"Conf: {conf}  LostFrames: {lost_frames}", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)

            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            pub_socket.send(base64.b64encode(buffer))
            raw.truncate(0)

    finally:
        print("Encerrando...")
        try:
            enviar_comando_motor(command_queue, 0, 0)
            arduino.write(b'a\n'); arduino.close()
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
