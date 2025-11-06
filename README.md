# Ligne-de-Janeiro

Projet de robot suiveur de ligne développé pour l'EI ST5 VAC 2025. Ce dépôt rassemble les codes de départ pour la détection de trajectoire, la prise de décision et le pilotage du robot au moyen d'un couple Raspberry Pi / Arduino.

## Contenu du dépôt

- `ARDUINO/` – Programmes Arduino, dont :
  - `serial_link/` pour la communication série avec le Raspberry Pi.
  - `test_*` pour vérifier individuellement les capteurs et actionneurs depuis l'IDE Arduino (veiller à câbler chaque capteur sur les broches indiquées dans le sketch correspondant).
- `basic_image_processing/` – Modules Python démontrant les traitements de vision utilisés pour détecter la ligne, les intersections et les panneaux. Ils constituent la base à étendre pour vos propres algorithmes.
- `basic_infrastructure/` – Code Python gérant la boucle principale, la communication réseau (ZeroMQ) et la coordination entre perception et décision.
- `basic_motion/` – Scripts Python illustrant l'échange de commandes entre le Raspberry Pi et l'Arduino (`dialogue.py`, `test_moteurs.py`, etc.). Ces scripts supposent que l'Arduino exécute `serial_link.ino`.
- `requirements.txt` – Liste des dépendances Python nécessaires au fonctionnement des modules fournis.

## Pré-requis matériels

- Raspberry Pi avec caméra compatible (PiCamera ou USB).
- Arduino (Uno ou équivalent) relié au Pi par USB.
- Capteurs infrarouges / optiques pour le suivi de ligne.
- Actionneurs moteurs pilotables par l'Arduino.

## Mise en place de l'environnement Python

```bash
# (Optionnel) création d'un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

### Dépendances principales

- **OpenCV (`opencv-python`)** – Traitement d'image en temps réel.
- **NumPy (`numpy`)** – Calcul matriciel et manipulation de tableaux.
- **PySerial (`pyserial`)** – Communication série entre Raspberry Pi et Arduino.
- **PyZMQ (`pyzmq`)** – Transport de messages réseau utilisé par l'infrastructure logicielle.

> **Remarque :** La bibliothèque `picamera` n'est pas incluse par défaut car elle ne fonctionne que sur Raspberry Pi/Linux. Sous macOS ou Windows, les scripts nécessitant la caméra devront être adaptés (utiliser par exemple une webcam USB via OpenCV).

## Organisation logicielle

### Traitement d'image (`basic_image_processing`)

- Détection de la ligne et des intersections via OpenCV.
- Extraction de métriques (position de la ligne, classification des noeuds L/T/+).
- Modules prêts à être enrichis avec vos propres heuristiques ou modèles.

### Infrastructure (`basic_infrastructure`)

- Boucle principale qui collecte les informations de vision et décide de la manœuvre.
- Gestion d'un système mémoire pour stabiliser les détections d'intersections.
- Publication des commandes via ZeroMQ.

### Mouvement (`basic_motion`)

- Scripts de test des moteurs et de dialogue série.
- Exemple de protocole de commande entre le Raspberry Pi et l'Arduino.

## Flux de fonctionnement type

1. **Acquisition** – La caméra du Raspberry Pi capture les images de la piste.
2. **Perception** – Les modules de `basic_image_processing` détectent la ligne et identifient les intersections pertinentes.
3. **Décision** – `basic_infrastructure` combine les informations perçues avec l'état courant du robot pour choisir la prochaine action (continuer, tourner, s'arrêter).
4. **Action** – Les commandes générées sont envoyées à l'Arduino via `basic_motion`, qui pilote les moteurs.

## Tests et validation

- Utiliser les sketches `ARDUINO/test_*` pour vérifier chaque capteur et actionneur indépendamment.
- Sur le Raspberry Pi, lancer les scripts de `basic_motion` pour valider la communication série (`python dialogue.py`, `python test_moteurs.py`).
- Pour valider la cohérence du code Python sans exécution sur le robot, une compilation statique simple peut être réalisée :

  ```bash
  python -m compileall basic_infrastructure
  ```

## Bonnes pratiques

- Conserver des versions fonctionnelles des fichiers clés afin de pouvoir revenir rapidement en arrière en cas de régression.
- Documenter les paramètres expérimentés (vitesses moteur, seuils de détection, etc.) pour faciliter la reproduction des résultats.
- Tester systématiquement sur piste réelle après chaque évolution majeure du code, notamment les ajustements des heuristiques de détection d'intersections.

## Ressources supplémentaires

- Se référer au document **« Doc robot - Codes et fichiers de départ.docx »** pour un complément d'explications détaillées sur l'architecture matérielle et logicielle.
- Consulter la documentation officielle d'OpenCV et de PySerial pour approfondir l'intégration vision/commande.

## Licence

Ce dépôt est fourni comme matériel pédagogique. Veuillez respecter les consignes de votre encadrant pour le partage ou la modification du code.
