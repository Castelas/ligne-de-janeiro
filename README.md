Codes de départ pour l'EI ST5 VAC 2025
(se référer au document 'Doc robot - Codes et fichiers de départ.docx' pour plus de détails)

* Les dossiers commençant par test_* contiennent des codes Arduino pour
	tester les capteurs et actionneurs directement depuis l'IDE Arduino
	(pour certains il faut d'abord brancher le capteur sur le(s) bon(s)
	pin(s) !)
* Les dossiers basic_image_processing et basic_infrastructure contiennent
	du code Python exemple destiné à être amélioré pour réaliser les
	fonctionnalités correspondantes du système.
* Le dossier basic_motion contient du code exemple permettant une
	interaction du Raspberry Pi avec l'Arduino. Les scripts Python
	test_moteurs.py et dialogue.py nécessitent de programmer l'Arduino
	avec le code serial_link.ino.

---

# ligne-de-janeiro
Project of line follower robot

## Setup do Ambiente

### Virtual Environment
Este projeto usa uma virtual environment Python. Para ativar:

```bash
# Ativar a venv
source venv/bin/activate

# Verificar se está ativa
which python  # deve mostrar caminho da venv

# Desativar quando terminar
deactivate
```

### Dependências
As bibliotecas necessárias já estão listadas em `requirements.txt` e instaladas na venv:

- **OpenCV** (`opencv-python`): Processamento de imagem e visão computacional
- **NumPy** (`numpy`): Computação numérica e arrays
- **PySerial** (`pyserial`): Comunicação serial com Arduino
- **PyZMQ** (`pyzmq`): Comunicação de rede (ZeroMQ)

### Nota sobre PiCamera
A biblioteca `picamera` não está incluída pois só funciona no Raspberry Pi/Linux. No macOS, os scripts que usam câmera não funcionarão, mas o resto do código pode ser testado.