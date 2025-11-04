# robot.py (versão final com modo manual/automático)
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
SERVER_IP = "192.168.137.22"  # <--- MUDE PARA O IP DO SEU SERVIDOR
MY_ID = 'bot001'

# --- PARÂMETROS DE VISÃO ---
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 200

# --- PARÂMETROS DE CONTROLE ---
VELOCIDADE_BASE = 150
VELOCIDADE_CURVA = 100
Kp = 0.8
VELOCIDADE_MAX = 255
MODO_AUTO = "AUTOMATICO"
MODO_MANUAL = "MANUAL"

# --- PARÂMETROS DE COMUNICAÇÃO SERIAL ---
PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200
# ###########################################################################

# Funções de detecção de interseções (adicionadas para visualização)
def line_intersection(line1, line2):
    """Calculate intersection point of two lines."""
    rho1, theta1 = line1
    rho2, theta2 = line2

    # Convert to cartesian coordinates
    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)

    # Check if lines are parallel
    if abs(theta1 - theta2) < 0.01 or abs(abs(theta1 - theta2) - np.pi) < 0.01:
        return None

    # Calculate intersection
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None

    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det

    return (int(x), int(y))

def merge_similar_lines(lines, rho_threshold=30, theta_threshold=np.pi/18):
    """Merge lines that are very close to each other."""
    if lines is None or len(lines) == 0:
        return []

    lines = lines.reshape(-1, 2)
    merged = []

    for rho, theta in lines:
        found_similar = False

        for i, (existing_rho, existing_theta) in enumerate(merged):
            rho_diff = abs(rho - existing_rho)
            theta_diff = min(abs(theta - existing_theta),
                           abs(abs(theta - existing_theta) - np.pi))

            if rho_diff < rho_threshold and theta_diff < theta_threshold:
                # Merge with existing line
                avg_rho = (existing_rho + rho) / 2
                avg_theta = (existing_theta + theta) / 2
                merged[i] = (avg_rho, avg_theta)
                found_similar = True
                break

        if not found_similar:
            merged.append((rho, theta))

    return merged

def detect_line_intersections(dilated_mask):
    """Detect line intersections using Hough Transform."""
    # Apply edge detection
    edges = cv2.Canny(dilated_mask, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return [], []

    # Merge similar lines
    unique_lines = merge_similar_lines(lines)

    # Find intersections
    intersection_points = []
    height, width = dilated_mask.shape[:2]

    for line1, line2 in combinations(unique_lines, 2):
        point = line_intersection(line1, line2)
        if point and 0 <= point[0] < width and 0 <= point[1] < height:
            intersection_points.append(point)

    # Filter duplicate points
    if intersection_points:
        filtered = []
        for i, pt1 in enumerate(intersection_points):
            cluster = [pt1]
            for j, pt2 in enumerate(intersection_points[i+1:], i+1):
                if np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2) < 30:
                    cluster.append(pt2)
            avg_x = int(np.mean([pt[0] for pt in cluster]))
            avg_y = int(np.mean([pt[1] for pt in cluster]))
            filtered.append((avg_x, avg_y))
        intersection_points = filtered

    return intersection_points, unique_lines

# ###########################################################################

# Funções de processamento de imagem e controle (do código anterior)
def processar_imagem(imagem):
    h, w = imagem.shape[:2]
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    eroded = cv2.erode(thresh, None, iterations=1)
    dilated = cv2.dilate(eroded, None, iterations=1)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx = w // 2; erro = 0
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c);
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cv2.drawContours(imagem, [c], -1, (0, 255, 0), 2)
            cv2.circle(imagem, (cx, int(M["m01"] / M["m00"])), 7, (0, 0, 255), -1)
    erro = cx - (w // 2)
    return imagem, erro

def calcular_velocidades_auto(erro):
    correcao = Kp * erro
    v_esq = np.clip(VELOCIDADE_BASE + correcao, 100, VELOCIDADE_MAX)
    v_dir = np.clip(VELOCIDADE_BASE - correcao, 100, VELOCIDADE_MAX)
    return int(v_esq), int(v_dir)

def enviar_comando_motor_serial(arduino, v_esq, v_dir):
    comando = f"C {v_dir} {v_esq}\n"
    arduino.write(comando.encode('utf-8'))

def main():
    # --- INICIALIZAÇÕES ---
    context = zmq.Context()
    # Socket PUB para vídeo
    pub_socket = context.socket(zmq.PUB); pub_socket.bind('tcp://*:5555')
    # Socket REQ para comandos
    req_socket = context.socket(zmq.REQ); req_socket.connect(f"tcp://{SERVER_IP}:5005")
    # Câmera
    camera = PiCamera(); camera.resolution = (IMG_WIDTH, IMG_HEIGHT); camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT)); time.sleep(0.1)
    # Arduino
    arduino = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=1); time.sleep(2)
    arduino.write(b'A10\n')
    print(f"Arduino respondeu: {arduino.readline().decode('utf-8').strip()}")

    # --- LÓGICA DE ESTADO ---
    current_mode = MODO_AUTO # Começa no modo automático
    v_esq, v_dir = 0, 0

    print("Robo iniciado. Pressione 'm' no controle para trocar de modo.")
    
    try:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array

            # 1. VERIFICAR COMANDOS DA REDE
            req_socket.send_pyobj({"from": MY_ID, "cmd": "key_request"})
            reply = req_socket.recv_pyobj()
            key = reply.get("key", "")
            
            if key == 'm':
                current_mode = MODO_MANUAL if current_mode == MODO_AUTO else MODO_AUTO
                print(f"Modo alterado para: {current_mode}")
                v_esq, v_dir = 0, 0 # Para o robô ao trocar de modo
            
            # 2. EXECUTAR LÓGICA DO MODO ATUAL
            if current_mode == MODO_AUTO:
                image, erro = processar_imagem(image)
                v_esq, v_dir = calcular_velocidades_auto(erro)
            elif current_mode == MODO_MANUAL:
                if key == 'w': v_esq, v_dir = VELOCIDADE_BASE, VELOCIDADE_BASE
                elif key == 's': v_esq, v_dir = -VELOCIDADE_BASE, -VELOCIDADE_BASE
                elif key == 'a': v_esq, v_dir = -VELOCIDADE_CURVA, VELOCIDADE_CURVA
                elif key == 'd': v_esq, v_dir = VELOCIDADE_CURVA, -VELOCIDADE_CURVA
                # Se não for uma tecla de movimento, mas estiver no modo manual, para
                elif key != '': v_esq, v_dir = 0, 0

            # 3. ENVIAR COMANDOS PARA O ARDUINO
            enviar_comando_motor_serial(arduino, v_esq, v_dir)

            # 4. PREPARAR E ENVIAR VÍDEO COM DETECÇÃO VISUAL
            # Processar imagem para detecção de interseções
            display_frame = image.copy()

            # Detect intersections and add visual elements
            # Process mask for intersection detection
            blurred = cv2.GaussianBlur(image, (7, 7), 0)
            _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
            hsv = cv2.cvtColor(binary, cv2.COLOR_BGR2HSV)

            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)

            kernel = np.ones((8, 8), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=2)
            dilated = cv2.dilate(eroded, kernel, iterations=2)

            # Detect intersections
            intersections, detected_lines = detect_line_intersections(dilated)

            # Draw processed mask overlay (semi-transparent)
            mask_colored = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            mask_colored = cv2.applyColorMap(mask_colored, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_colored, 0.3, 0)

            # Draw detected lines (green like corner_detection.py)
            if detected_lines:
                for rho, theta in detected_lines:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw intersection points (red with numbers)
            if intersections:
                for idx, (x, y) in enumerate(intersections, 1):
                    cv2.circle(display_frame, (x, y), 8, (0, 0, 255), -1)
                    cv2.circle(display_frame, (x, y), 12, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"{idx}", (x+15, y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add status text
            cv2.putText(display_frame, f"Modo: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"V_E:{v_esq} V_D:{v_dir}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(display_frame, f"Lines: {len(detected_lines)} Intersections: {len(intersections)}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            # Comprimir e enviar
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            pub_socket.send(base64.b64encode(buffer))
            
            rawCapture.truncate(0)

    finally:
        print("Encerrando...")
        enviar_comando_motor_serial(arduino, 0, 0) # Para os motores!
        arduino.write(b'a\n')
        arduino.close()
        pub_socket.close(); req_socket.close(); context.term()

if __name__ == "__main__":
    main()
