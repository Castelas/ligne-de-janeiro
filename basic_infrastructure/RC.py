# RC.py - Carrinho de Controle Remoto
import zmq
import serial
import time
import cv2
import base64
from picamera import PiCamera
from picamera.array import PiRGBArray

# --- PARÂMETROS ---
SERVER_IP = "192.168.137.164"  # IP do computador que roda o server.py
VELOCIDADE_BASE = 120          # Velocidade base para movimento
IMG_WIDTH, IMG_HEIGHT = 640, 480

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

def init_control_connection():
    """Inicializa conexão com servidor de controle"""
    context = zmq.Context()
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(f"tcp://{SERVER_IP}:5005")
    req_socket.setsockopt(zmq.RCVTIMEO, 1000)  # Timeout 1s
    return req_socket, context

def init_camera_and_stream():
    """Inicializa câmera e stream ZMQ"""
    try:
        camera = PiCamera()
        camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
        camera.framerate = 24
        time.sleep(1.0)  # Warm-up

        # Inicializar stream ZMQ
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5555")
        print("📹 Stream ZMQ inicializado em tcp://*:5555")

        return camera, socket, context
    except Exception as e:
        print(f"Erro ao inicializar câmera/stream: {e}")
        return None, None, None

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
        if key:
            print(f"📥 Recebido comando: {key}")
        return key if key else None
    except zmq.Again:
        # Timeout normal, não imprime
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

    camera, stream_socket, stream_context = init_camera_and_stream()
    if not camera:
        print("❌ Não foi possível inicializar câmera. Saindo...")
        return

    req_socket, control_context = init_control_connection()

    print("🚗 Carrinho RC iniciado!")
    print("Aguardando comandos do controle remoto...")
    print("Pressione Ctrl+C para sair.")

    current_key = ''  # Última tecla enviada

    try:
        # Loop de captura e transmissão de vídeo
        raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
        for frame in camera.capture_continuous(raw, format="bgr", use_video_port=True):
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

            # Transmitir frame
            img = frame.array
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            stream_socket.send(base64.b64encode(buffer))

            raw.truncate(0)

            time.sleep(0.05)  # Controlar frame rate

    except KeyboardInterrupt:
        print("\nSaindo...")
    finally:
        print("Encerrando...")
        drive_cap(arduino, 0, 0)  # Parar o carrinho
        if arduino:
            arduino.close()
        if camera:
            camera.close()
        if stream_socket:
            stream_socket.close()
        if stream_context:
            stream_context.term()
        req_socket.close()
        control_context.term()

if __name__ == "__main__":
    main()
