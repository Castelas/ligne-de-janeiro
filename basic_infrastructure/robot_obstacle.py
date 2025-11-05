# robot2.py — seguidor de linha com ROI, confiança e estado FOLLOW/LOST (sem derivativo)

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
MY_ID     = "bot001"              # identificador do robô na rede

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

# ======================== OBSTACLE/ULTRASSOM CONFIG ==========================
REVERSE_TIME_S = 1.5     # tempo de ré quando obstáculo detectado
SPIN_TIME_S    = 0.8     # tempo aproximado para ~180° (ajuste no seu robô)
ULTRA_THRESHOLD_CM = 25  # distância para considerar obstáculo

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

def enviar_comando_motor_serial(arduino, v_esq, v_dir):
    # Envia velocidades com sinal; negativos significam ré
    comando = f"C {v_dir} {v_esq}\n"
    arduino.write(comando.encode('utf-8'))

def ler_obstaculo_serial(arduino):
    """Retorna True quando o firmware envia 'OB' (não bloqueante)."""
    try:
        # lê todas as linhas disponíveis no buffer (não bloqueante)
        while getattr(arduino, "in_waiting", 0) > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            if line.upper().startswith("OB"):
                return True
        return False
    except Exception:
        return False


def rotina_obstaculo(arduino):
    # 1) parar
    arduino.write(b"C 0 0\n"); time.sleep(0.05)
    # 2) ré
    arduino.write(b"C -160 -160\n"); time.sleep(REVERSE_TIME_S)
    # 3) giro ~180°
    arduino.write(b"C 170 -170\n"); time.sleep(SPIN_TIME_S)
    # 4) parar e rearmar proteção
    arduino.write(b"C 0 0\n"); time.sleep(0.1)
    try:
        arduino.write(b'I1\n')
    except Exception:
        pass

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
        # habilita proteção de obstáculo no firmware (para emitir 'OB')
        arduino.write(b'I1\n')
    except Exception:
        pass

    # Estado do seguidor (sem derivativo)
    last_err = 0.0
    lost_frames = 0
    state = 'FOLLOW'  # FOLLOW | LOST | OBSTACLE

    current_mode = MODO_AUTO
    v_esq, v_dir = 0, 0
    print("Robo iniciado. Pressione 'm' para trocar de modo.")

    try:
        state = ""            # estado exibido no overlay (ex: 'FOLLOW', 'LOST', 'OBSTACLE')
        current_mode = MODO_AUTO if True else MODO_MANUAL  # mantenha sua lógica real aqui
        for frame in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            image = frame.array
            # processamento da imagem (erro, conf)
            frame_annot, erro, conf = processar_imagem(image.copy())

            # verifica obstáculo vindo do Arduino (serial)
            if ler_obstaculo_serial(arduino):
                state = 'OBSTACLE'
                print("OBSTACLE detectado (via serial)")
                # executa rotina de evasão (no Arduino ou local conforme implementado)
                rotina_obstaculo(arduino)

            # desenha mode/state no overlay da câmera
            cv2.putText(frame_annot, f"MODE: {current_mode}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame_annot, f"STATE: {state}", (8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # publica/mostra frame conforme sua pipeline existente
            # ex: pub_socket.send_pyobj(frame_annot)
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
