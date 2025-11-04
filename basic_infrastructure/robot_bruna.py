# robot.py (versão final com modo manual/automático)
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import zmq
import base64
import time
import numpy as np
import serial

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

            # 4. PREPARAR E ENVIAR VÍDEO
            # Adicionar texto de status no vídeo
            cv2.putText(image, f"Modo: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"V_E:{v_esq} V_D:{v_dir}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            # Comprimir e enviar
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
