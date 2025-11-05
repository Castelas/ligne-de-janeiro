# control.py (versão final)
import cv2
import zmq
import base64
import numpy as np

# --- PARÂMETROS ---
SERVER_IP = "192.168.137.176"  # <--- MUDE PARA O IP DO SEU SERVIDOR
ROBOT_IP = "192.168.137.69"   # <--- MUDE PARA O IP DO SEU ROBÔ (RASPBERRY PI)
# --- FIM DOS PARÂMETROS ---

def main():
    context = zmq.Context()
    # Socket SUB para vídeo
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://{ROBOT_IP}:5555")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
    # Socket REQ para comandos
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(f"tcp://{SERVER_IP}:5005")
    
    print("Controle iniciado.")
    print("Use W,A,S,D para controle manual.")
    print("Pressione 'm' para alternar entre modo MANUAL e AUTOMATICO.")
    print("Pressione 'q' para sair.")
    
    try:
        while True:
            # Recebe e exibe o quadro de vídeo
            frame_b64 = sub_socket.recv()
            img_buffer = base64.b64decode(frame_b64)
            frame = cv2.imdecode(np.frombuffer(img_buffer, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("Controle do Robo", frame)

            # Verifica se uma tecla foi pressionada
            key = cv2.waitKey(1) & 0xFF
            
            if key != 255: # Se uma tecla foi pressionada
                char_key = chr(key)
                
                if char_key == 'q':
                    break # Sai do loop
                
                # Envia qualquer tecla pressionada (w,a,s,d,m, etc.) para o servidor
                msg = {"from": "control", "cmd": "key", "key": char_key}
                req_socket.send_pyobj(msg)
                req_socket.recv_pyobj() # Espera confirmação
                
    finally:
        print("Encerrando...")
        cv2.destroyAllWindows()
        sub_socket.close(); req_socket.close(); context.term()

if __name__ == "__main__":
    main()