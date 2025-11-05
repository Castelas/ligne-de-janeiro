# robot_new.py — seguidor de linha com rota automática pré-definida
# Lógica de estado: FOLLOW, LOST, APPROACHING, STOPPING, STOPPED (automático),
#                  TURN_LEFT, TURN_RIGHT, GO_STRAIGHT

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
SERVER_IP = "192.168.137.78"     # <--- COLOQUE o IP do servidor (control.py)
MY_ID     = "bot001"             # identificador do robô na rede

# --- VISÃO ---
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 180

# Hough (apenas para overlay/depuração visual — NÃO usado no controle)
HOUGHP_THRESHOLD    = 35
HOUGHP_MINLEN_FRAC  = 0.35
HOUGHP_MAXGAP       = 20
ROI_CROP_FRAC       = 0.10  # zera 20% do topo para reduzir ruído no overlay
RHO_MERGE           = 40
THETA_MERGE_DEG     = 6
ORTH_TOL_DEG        = 15
PAR_TOL_DEG         = 8

# --- CONTROLE (P puro; sem derivativo ainda) ---
VELOCIDADE_BASE = 110
VELOCIDADE_CURVA = 100
Kp = 0.8
VELOCIDADE_MAX = 255
MODO_AUTO   = "AUTOMATICO"
MODO_MANUAL = "MANUAL"

# Ajustes (detecção/recuperação)
E_MAX_PIX       = IMG_WIDTH // 2        # erro máximo usado para escalonar velocidade
V_MIN           = 0                     # velocidade mínima admitida no AUTO
SEARCH_SPEED    = 120                   # velocidade para girar no lugar em LOST
LOST_MAX_FRAMES = 8                     # frames sem confiança até entrar em LOST
DEAD_BAND       = 6                     # |erro| <= DEAD_BAND => erro = 0
ROI_BOTTOM_FRAC = 0.55                  # início da ROI inferior (55% da altura)
MIN_AREA_FRAC   = 0.004                 # área mínima do contorno na ROI (fração)
MAX_AREA_FRAC   = 0.25                  # área máxima aceitável (descarta “piso inteiro”)
ASPECT_MIN      = 2.0                   # formato “faixa”: comprimento/largura mínimo
LINE_POLARITY   = 'auto'                # 'white', 'black' ou 'auto'
USE_ADAPTIVE    = False                 # threshold adaptativo desligado por padrão

# NOVOS PARÂMETROS PARA INTERSEÇÃO
Y_START_SLOWING_FRAC = 0.50  # Começa a frear quando a interseção passa de 70% da altura
Y_TARGET_STOP_FRAC = 0.95    # Ponto de parada (para iniciar o "crawl") a 95% da altura
CRAWL_SPEED = 80             # Velocidade baixa para o "anda mais um pouco"
CRAWL_DURATION_S = 0.1       # Duração (segundos) do "anda mais um pouco"

# NOVOS PARÂMETROS PARA AÇÕES NA INTERSEÇÃO
TURN_SPEED = 130             # Velocidade para girar (90 graus)
TURN_DURATION_S = 1          # Duração (segundos) para o giro (AJUSTAR NA PRÁTICA)
STRAIGHT_SPEED = 90         # Velocidade para "seguir reto"

TURN_DURATION_S = 1          # Duração (segundos) para o giro (AJUSTAR NA PRÁTICA)
STRAIGHT_SPEED = 100         # Velocidade para "seguir reto"
STRAIGHT_DURATION_S = 0.5    # Duração (segundos) para atravessar (AJUSTAR)

# =========================================================================
# === ROTA AUTOMÁTICA ===
# Defina a sequência de ações que o robô deve tomar nas interseções
# Comandos válidos: "RIGHT", "LEFT", "STRAIGHT"
# =========================================================================
ACTION_SEQUENCE = ["LEFT", "STRAIGHT", "STRAIGHT", "RIGHT"]
# =========================================================================


# --- SERIAL ---
PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200

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
    v_esq = int(np.clip(v_esq, -VELOCIDADE_MAX, VELOCIDADE_MAX))
    v_dir = int(np.clip(v_dir, -VELOCIDADE_MAX, VELOCIDADE_MAX))
    return v_esq, v_dir

def enviar_comando_motor_serial(arduino, v_esq, v_dir):
    # Envia velocidades com sinal; negativos significam ré
    comando = f"C {v_dir} {v_esq}\n"
    arduino.write(comando.encode('utf-8'))

# ================================= MAIN =====================================
def main():
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

    # --- Arduino ---
    arduino = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=1); time.sleep(2)
    try:
        arduino.write(b'A10\n')
        print(f"Arduino: {arduino.readline().decode('utf-8').strip()}")
    except Exception:
        pass

    # Estado do seguidor
    last_err = 0.0
    lost_frames = 0
    # ESTADOS: 'FOLLOW', 'LOST', 'APPROACHING', 'STOPPING', 'STOPPED',
    #          'TURN_LEFT', 'TURN_RIGHT', 'GO_STRAIGHT'
    state = 'FOLLOW'
    action_start_time = 0.0 # Generaliza o temporizador para todas as ações
    last_known_y = -1.0     # Última posição Y válida da interseção
    intersection_counter = 0  # Índice para ACTION_SEQUENCE

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
                state = 'FOLLOW'
                last_known_y = -1.0 # Reseta
                intersection_counter = 0 # Reseta a rota

            conf = 0  # default p/ HUD caso esteja no modo MANUAL
            intersections = []
            detected_lines_for_hud = []

            # ----------------- CONTROLE -----------------
            if current_mode == MODO_AUTO:
                # Processamento de imagem (linha e interseções)
                image, erro, conf = processar_imagem(image)
                
                display_frame_temp = image.copy()
                mask = build_binary_mask(display_frame_temp)
                intersections, detected_lines_for_hud = detect_intersections(mask)

                # Encontrar a interseção alvo (a mais próxima, com maior 'y')
                target_intersection = None
                target_y = -1
                if intersections:
                    intersections.sort(key=lambda p: p[1], reverse=True)
                    target_intersection = intersections[0]
                    target_y = target_intersection[1]

                h, w = image.shape[:2]
                Y_START_SLOWING = h * Y_START_SLOWING_FRAC
                Y_TARGET_STOP = h * Y_TARGET_STOP_FRAC

                # --- Máquina de Estados de Controle ---

                # 1. Transições de Estado
                if state == 'FOLLOW':
                    if conf == 0:
                        lost_frames += 1
                        if lost_frames >= LOST_MAX_FRAMES:
                            state = 'LOST'
                            last_known_y = -1.0 # Reseta
                    elif target_y > Y_START_SLOWING:
                        print("Interseção detectada! Iniciando aproximação.")
                        state = 'APPROACHING'
                        last_known_y = target_y # Armazena Y inicial
                    else:
                        # Tudo normal, sem interseção
                        lost_frames = 0
                        last_err = erro
                        last_known_y = -1.0 # Garante que está limpo

                elif state == 'APPROACHING':
                    # Prioridade 1: Perdemos a linha? (Condição de falha)
                    if conf == 0:
                        print("Linha perdida durante aproximação.")
                        state = 'LOST'
                        last_known_y = -1.0 # Reseta
                        
                    # Prioridade 2: Ainda vemos a interseção?
                    elif target_y != -1:
                         last_known_y = target_y # Atualiza a posição
                         
                         # GATILHO 1: Atingimos o alvo de Y?
                         if last_known_y >= Y_TARGET_STOP:
                            print("Alvo (Y_TARGET_STOP) atingido. 'Andando mais um pouco'...")
                            state = 'STOPPING'
                            action_start_time = time.time()
                            last_known_y = -1.0 # Reseta para a próxima
                         else:
                            # Se não atingimos, apenas continuamos seguindo a linha
                            last_err = erro 
                    
                    # Prioridade 3: A interseção DESAPARECEU (target_y == -1)
                    # MAS ainda vemos a LINHA (conf == 1)?
                    # (Gatilho 2: A câmera passou por cima da interseção)
                    elif target_y == -1 and conf == 1:
                        print("Interseção passou do FoV. 'Andando mais um pouco'...")
                        state = 'STOPPING'
                        action_start_time = time.time()
                        last_known_y = -1.0 # Reseta para a próxima

                    # Se nenhuma das anteriores, e ainda vemos a linha,
                    # apenas continuamos seguindo a linha.
                    elif conf == 1:
                        last_err = erro
                
                elif state == 'STOPPING':
                    if (time.time() - action_start_time) > CRAWL_DURATION_S:
                        print("Parada completa. Decidindo acao automatica...")
                        state = 'STOPPED' # Transição imediata para decisão
                
                elif state == 'STOPPED':
                    # Pega a próxima ação da sequência
                    
                    # Se a sequência terminou, reinicia do começo (loop)
                    if intersection_counter >= len(ACTION_SEQUENCE):
                        print("Fim da sequencia, reiniciando rota.")
                        intersection_counter = 0 
                    
                    command = ACTION_SEQUENCE[intersection_counter]
                    print(f"Intersecao #{intersection_counter}: Executando '{command}'")

                    # Muda para o estado de ação apropriado
                    if command == "LEFT":
                        state = 'TURN_LEFT'
                        action_start_time = time.time()
                    elif command == "RIGHT":
                        state = 'TURN_RIGHT'
                        action_start_time = time.time()
                    elif command == "STRAIGHT":
                        state = 'GO_STRAIGHT'
                        action_start_time = time.time()
                    else:
                        # Caso o comando na lista esteja errado, apenas segue em frente
                        print(f"Comando '{command}' desconhecido. Seguindo em frente.")
                        state = 'GO_STRAIGHT' 
                        action_start_time = time.time()
                    
                    # Incrementa o contador para a *próxima* interseção
                    intersection_counter += 1
                
                elif state == 'TURN_LEFT':
                    if (time.time() - action_start_time) > TURN_DURATION_S:
                        print("Giro completo. Procurando linha...")
                        state = 'FOLLOW'
                        last_err = -1 # Influencia a busca para a esquerda

                elif state == 'TURN_RIGHT':
                    if (time.time() - action_start_time) > TURN_DURATION_S:
                        print("Giro completo. Procurando linha...")
                        state = 'FOLLOW'
                        last_err = 1 # Influencia a busca para a direita

                elif state == 'GO_STRAIGHT':
                    if (time.time() - action_start_time) > STRAIGHT_DURATION_S:
                        print("Atravessou. Procurando linha...")
                        state = 'FOLLOW'

                elif state == 'LOST':
                    if conf == 1:
                        print("Linha reencontrada.")
                        state = 'FOLLOW'
                        lost_frames = 0
                        last_err = erro
                        last_known_y = -1.0 # Reseta

                # 2. Ações de Estado (Definir velocidades)
                if state == 'FOLLOW':
                    if conf == 1:
                        speed_scale = max(0.35, 1.0 - abs(erro) / float(E_MAX_PIX))
                        base_speed = int(np.clip(VELOCIDADE_BASE * speed_scale, V_MIN, VELOCIDADE_MAX))
                        v_esq, v_dir = calcular_velocidades_auto(erro, base_speed)
                    else:
                        base_speed = int(np.clip(VELOCIDADE_BASE * 0.35, V_MIN, VELOCIDADE_MAX))
                        v_esq, v_dir = calcular_velocidades_auto(0, base_speed)
                
                elif state == 'APPROACHING':
                    # USA last_known_y para o cálculo, não target_y
                    progress = 0.0
                    if (Y_TARGET_STOP - Y_START_SLOWING) > 0: # Evita divisão por zero
                        progress = (last_known_y - Y_START_SLOWING) / (Y_TARGET_STOP - Y_START_SLOWING)
                    
                    speed_factor = 1.0 - np.clip(progress, 0.0, 1.0)
                    
                    current_base_speed = (VELOCIDADE_BASE - CRAWL_SPEED) * speed_factor + CRAWL_SPEED
                    base_speed = int(np.clip(current_base_speed, CRAWL_SPEED, VELOCIDADE_MAX))
                    # Segue a linha (erro) mesmo durante a frenagem
                    v_esq, v_dir = calcular_velocidades_auto(erro, base_speed)

                elif state == 'STOPPING':
                    # "Anda mais um pouco" - crawl reto
                    v_esq, v_dir = CRAWL_SPEED, CRAWL_SPEED
                
                elif state == 'STOPPED':
                    # Este estado dura apenas 1 frame, então a velocidade é 0
                    # antes de mudar para TURN/STRAIGHT
                    v_esq, v_dir = 0, 0
                
                elif state == 'TURN_LEFT':
                    v_esq, v_dir = -TURN_SPEED, TURN_SPEED

                elif state == 'TURN_RIGHT':
                    v_esq, v_dir = TURN_SPEED, -TURN_SPEED

                elif state == 'GO_STRAIGHT':
                    v_esq, v_dir = STRAIGHT_SPEED, STRAIGHT_SPEED
                
                elif state == 'LOST':
                    # Lógica original de busca
                    turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                    v_esq, v_dir = int(turn), int(-turn)

            else:
                # MODO MANUAL (w,a,s,d reinicia o estado E A ROTA)
                if key == 'w':   
                    v_esq, v_dir = VELOCIDADE_BASE, VELOCIDADE_BASE
                    state = 'FOLLOW'
                    last_known_y = -1.0 # Reseta
                    intersection_counter = 0 # Reseta a rota
                elif key == 's': 
                    v_esq, v_dir = -VELOCIDADE_BASE, -VELOCIDADE_BASE
                    state = 'FOLLOW'
                    last_known_y = -1.0 # Reseta
                    intersection_counter = 0 # Reseta a rota
                elif key == 'a': 
                    v_esq, v_dir = -VELOCIDADE_CURVA, VELOCIDADE_CURVA
                    state = 'FOLLOW'
                    last_known_y = -1.0 # Reseta
                    intersection_counter = 0 # Reseta a rota
                elif key == 'd': 
                    v_esq, v_dir = VELOCIDADE_CURVA, -VELOCIDADE_CURVA
                    state = 'FOLLOW'
                    last_known_y = -1.0 # Reseta
                    intersection_counter = 0 # Reseta a rota
                elif key != '':
                    v_esq, v_dir = 0, 0

            enviar_comando_motor_serial(arduino, v_esq, v_dir)

            # ---------------- VISUALIZAÇÃO ----------------
            display_frame = image.copy()
            
            if current_mode != MODO_AUTO:
                mask = build_binary_mask(display_frame)
                intersections, detected_lines_for_hud = detect_intersections(mask)

            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)
            
            # desenha linhas (verde)
            for rho, theta in detected_lines_for_hud:
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

            # Atualiza o HUD com o novo estado
            cv2.putText(display_frame, f"Modo: {current_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"V_E:{v_esq} V_D:{v_dir}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            
            # Texto do estado (com cor)
            state_color = (0, 255, 0) # Verde para FOLLOW
            if state == 'LOST': state_color = (0, 0, 255) # Vermelho
            elif state == 'APPROACHING': state_color = (0, 255, 255) # Amarelo
            elif state == 'STOPPING': state_color = (255, 0, 255) # Magenta
            elif state == 'STOPPED': state_color = (255, 0, 0) # Azul
            elif state in ['TURN_LEFT', 'TURN_RIGHT', 'GO_STRAIGHT']: state_color = (255, 165, 0) # Laranja
            
            cv2.putText(display_frame, f"State: {state}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            # Adiciona prompt de Ação Automática
            if state == 'STOPPING':
                 next_action_index = intersection_counter
                 if next_action_index >= len(ACTION_SEQUENCE):
                     next_action_index = 0 # Mostra que vai reiniciar
                 
                 next_action = ACTION_SEQUENCE[next_action_index]
                 
                 cv2.putText(display_frame, f"Parando. Prox: {next_action}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif state in ['TURN_LEFT', 'TURN_RIGHT', 'GO_STRAIGHT']:
                 cv2.putText(display_frame, f"Acao: {state}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            cv2.putText(display_frame, f"Lines: {len(detected_lines_for_hud)}", (10, 105),
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
