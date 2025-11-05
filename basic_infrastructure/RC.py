# RC.py - Carrinho de Controle Remoto
import cv2
import zmq
import base64
import numpy as np
import argparse
import serial
import time

# --- PARÂMETROS ---
SERVER_IP = "192.168.137.164"  # IP do computador que roda o server.py
ROBOT_IP = "192.168.137.69"    # IP do Raspberry Pi
VELOCIDADE_BASE = 120          # Velocidade base para movimento

# --- FUNÇÕES ---

def init_serial():
    """Inicializa conexão serial com Arduino"""
    try:
        arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        time.sleep(2)  # Espera estabilizar
        print("Arduino conectado.")
        return arduino
    except Exception as e:
        print(f"Erro ao conectar Arduino: {e}")
        return None

def drive_cap(arduino, left, right, cap=255):
    """Envia comando de movimento para Arduino"""
    if arduino:
        left = max(-cap, min(cap, left))
        right = max(-cap, min(cap, right))
        cmd = f"{left},{right}\n"
        arduino.write(cmd.encode())

def init_video_stream():
    """Inicializa stream de vídeo"""
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://{ROBOT_IP}:5555")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
    return sub_socket, context

def init_control_connection():
    """Inicializa conexão com servidor de controle"""
    context = zmq.Context()
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(f"tcp://{SERVER_IP}:5005")
    req_socket.setsockopt(zmq.RCVTIMEO, 1000)  # Timeout 1s
    return req_socket, context

def manual_control(arduino, key):
    """Controle manual baseado na tecla pressionada"""
    if key == 'w':  # Frente
        drive_cap(arduino, VELOCIDADE_BASE, VELOCIDADE_BASE)
        print("🕹️  RC: Frente")
    elif key == 's':  # Trás
        drive_cap(arduino, -VELOCIDADE_BASE, -VELOCIDADE_BASE)
        print("🕹️  RC: Trás")
    elif key == 'a':  # Esquerda
        drive_cap(arduino, -VELOCIDADE_BASE, VELOCIDADE_BASE)
        print("🕹️  RC: Girando esquerda")
    elif key == 'd':  # Direita
        drive_cap(arduino, VELOCIDADE_BASE, -VELOCIDADE_BASE)
        print("🕹️  RC: Girando direita")
    elif key == 'stop':  # Parar
        drive_cap(arduino, 0, 0)
        print("🕹️  RC: Parado")

def get_remote_key(req_socket):
    """Obtém tecla do controle remoto"""
    try:
        msg = {"from": "robot", "cmd": "key_request"}
        req_socket.send_pyobj(msg)
        reply = req_socket.recv_pyobj()
        key = reply.get("key", "")
        return key if key else None
    except zmq.Again:
        return None
    except Exception as e:
        print(f"⚠️  Erro ao obter tecla remota: {e}")
        return None

def main():
    """Função principal do carrinho RC"""
    # Inicializar conexões
    arduino = init_serial()
    if not arduino:
        print("❌ Não foi possível conectar ao Arduino. Saindo...")
        return

    sub_socket, video_context = init_video_stream()
    req_socket, control_context = init_control_connection()

    print("🚗 Carrinho RC iniciado!")
    print("Use W/A/S/D para controlar (segurar as teclas).")
    print("Pressione 'q' para sair.")
    print("Quando soltar as teclas, o carrinho para automaticamente.")

    current_key = ''  # Última tecla enviada

    try:
        while True:
            # Recebe e exibe o quadro de vídeo
            try:
                frame_b64 = sub_socket.recv(flags=zmq.NOBLOCK)
                img_buffer = base64.b64decode(frame_b64)
                frame = cv2.imdecode(np.frombuffer(img_buffer, np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow("Carrinho RC", frame)
            except zmq.Again:
                pass  # Nenhum frame novo

            # Obtém comando remoto
            remote_key = get_remote_key(req_socket)

            if remote_key in ['w', 'a', 's', 'd']:
                # Movimento
                if remote_key != current_key:
                    manual_control(arduino, remote_key)
                    current_key = remote_key
            elif remote_key == 'stop' or (current_key and not remote_key):
                # Parar
                manual_control(arduino, 'stop')
                current_key = ''

            # Verifica se 'q' foi pressionado na janela de vídeo
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Encerrando...")
        drive_cap(arduino, 0, 0)  # Parar o carrinho
        if arduino:
            arduino.close()
        cv2.destroyAllWindows()
        sub_socket.close()
        req_socket.close()
        video_context.term()
        control_context.term()

if __name__ == "__main__":
    main()
