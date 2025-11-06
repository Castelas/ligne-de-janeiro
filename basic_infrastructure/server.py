# server.py (sem alterações)
import zmq
import sys

# ---> MUDE AQUI PARA O IP DO COMPUTADOR QUE RODA ESTE SERVIDOR <---
server_ip = "192.168.137.22" 

verbose_mode = True
nodes = list()
global key 
key = '' 

def main():
    repsock = create_connection_interface(server_ip)
    print(f"Servidor iniciado em {server_ip}. Aguardando conexoes...")
    while True:
        msg = repsock.recv_pyobj()
        reply = process_msg(msg)
        repsock.send_pyobj(reply)
        if verbose_mode:
            print(f"Recebido: {msg} -> Respondido: {reply}")

def process_msg(msg):
    global key
    default_reply = {"status": "ok"}
    msg_type = msg.get("cmd", "")
    msg_from = msg.get("from", "")

    if msg_type == "log":
        if msg_from not in nodes:
            print(f"Novo no '{msg_from}' se registrando.")
            nodes.append(msg_from)
        reply = default_reply
    elif msg_from == "control" and msg_type == "key":
        key = msg.get("key", '')
        print(f"🎮 Recebido comando do control: {key}")
        reply = default_reply
    elif msg_from != "control" and msg_type == "key_request":
        reply = {"key": key}
        if key:
            print(f"🤖 Enviando comando para robot: {key}")
        key = '' # Limpa a tecla após ser lida pelo robô
    else:
        reply = {"error": "mensagem invalida"}
    return reply

def create_connection_interface(ip):
    ctx = zmq.Context()
    repsock = ctx.socket(zmq.REP)
    repaddr = f"tcp://{ip}:5005"
    repsock.bind(repaddr)
    return repsock

if __name__ == "__main__":
    main()
