# delivery_delivery_v12.py — Delivery 4x4 (robot2 vision/control inline)
# Novidades v12:
#  • Pós-pivô em 2 fases: (a) girar até VER a linha; (b) avançar devagar usando P até CENTRALIZAR.
#    Isso resolve o “não anda o suficiente depois do pivô” e melhora o lock.
#  • Intersecções menos “estritas”: banda mais baixa e estabilidade em 2 frames.
#  • Mantém todas as rotinas de visão/controle idênticas ao robot2.py.
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2, time, numpy as np, serial, argparse, zmq, base64

# --- CONFIGURAÇÃO CONTROLE REMOTO ---
SERVER_IP = "192.168.137.164"  # IP do computador que roda o server.py
# --- FIM CONFIGURAÇÃO ---

# ============================= PARÂMETROS (iguais ao robot2) =============================
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 160  # Ajuste fino para melhor detecção
HOUGHP_THRESHOLD    = 35
HOUGHP_MINLEN_FRAC  = 0.35
HOUGHP_MAXGAP       = 20
ROI_CROP_FRAC       = 0.20
RHO_MERGE           = 40
THETA_MERGE_DEG     = 6
ORTH_TOL_DEG        = 15
PAR_TOL_DEG         = 8

VELOCIDADE_BASE = 100
VELOCIDADE_CURVA = 100
Kp = 1.2              # Ganho do controlador P - aumentado para melhor controle
VELOCIDADE_MAX = 255
E_MAX_PIX       = IMG_WIDTH // 2
V_MIN           = 0
SEARCH_SPEED    = 100
LOST_MAX_FRAMES = 5
DEAD_BAND       = 3
ROI_BOTTOM_FRAC = 0.55
MIN_AREA_FRAC   = 0.003  # Ainda mais reduzido para detectar linhas
MAX_AREA_FRAC   = 0.3    # Um pouco mais tolerante
ASPECT_MIN      = 2.0    # Ainda menos rigoroso
LINE_POLARITY   = 'white'               # Forçado para branco novamente
USE_ADAPTIVE    = False

PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200

# ======== DELIVERY (extra) ========
GRID_NODES = (5, 5)       # 4x4 quadrados → 5x5 nós
START_SPEED  = 90        # reta cega
TURN_SPEED   = 190        # giros 90/180 (mais rápidos)

# PIVÔ e aquisição pós-pivô
PIVOT_CAP       = 180     # limite superior do pivô - aumentado
PIVOT_MIN       = 150     # mínimo para vencer atrito - aumentado
PIVOT_TIMEOUT   = 1   # Ligeiramente aumentado para virar um tiquinho mais
SEEN_FRAMES     = 1       # frames consecutivos "vendo" a linha para sair do giro - reduzido
ALIGN_BASE      = 90      # velocidade base na fase de alinhamento (P)  [aumentada para mover o carrinho]
ALIGN_CAP       = 120     # cap de segurança na fase de alinhamento [reduzido]
ALIGN_TOL_PIX   = 8       # centralização final
ALIGN_STABLE    = 2       # frames estáveis [reduzido para entrar em FOLLOW mais rápido]
ALIGN_TIMEOUT   = 6.0     # tempo máx. alinhando (s) [aumentado significativamente]

# Intersecção (parâmetros do robot_pedro.py - mais robustos)
Y_START_SLOWING_FRAC = 0.60  # Começa a frear quando a interseção passa de 70% da altura
Y_TARGET_STOP_FRAC = 1.0     # Aumentado para 100% - passa completamente pela interseção
CRAWL_SPEED = 100            # Velocidade baixa para o "anda mais um pouco"
CRAWL_DURATION_S = 0.4       # Duração (segundos) do "anda mais um pouco" - aumentado
TURN_SPEED = 180             # Velocidade para girar (90 graus) - aumentado para giros mais precisos
TURN_DURATION_S = 0.75       # Duração (segundos) para o giro - ajustado para 0.75s
STRAIGHT_SPEED = 130         # Velocidade para "seguir reto"
STRAIGHT_DURATION_S = 0.5    # Duração (segundos) para atravessar

# Início cego (linha horizontal)
ROW_BAND_TOP_FRAC       = 0.45
ROW_BAND_BOTTOM_FRAC    = 0.85
ROW_PEAK_FRAC_THR       = 0.030
LOSE_FRAMES_START       = 5
START_TIMEOUT_S         = 6.0

# ============================ VISÃO (robot2) ============================
def _angle_diff(a, b):
    d = abs((a - b) % np.pi)
    return min(d, np.pi - d)

def _deg(x): return np.deg2rad(x)

def distance_to_line(point, line):
    """Calcula a distância de um ponto (x,y) a uma linha (rho, theta)"""
    rho, theta = line
    x, y = point
    return abs(x * np.cos(theta) + y * np.sin(theta) - rho)

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
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # Operações morfológicas simples
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove parte superior da imagem (céu/ruído)
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

def processar_imagem(imagem):
    global initial_frames_ignored

    h, w = imagem.shape[:2]
    cx_img = w // 2

    # Ignora detecções iniciais para evitar detectar chão/ruído
    initial_frames_ignored += 1
    if initial_frames_ignored <= IGNORE_INITIAL_FRAMES:
        return imagem, 0, 0  # Retorna sem detecção

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if USE_ADAPTIVE:
        th_white = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -5)
        th_black = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, -5)
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
            if area < roi_area * MIN_AREA_FRAC:  continue
            if area > roi_area * MAX_AREA_FRAC:  continue
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            if rw < 1 or rh < 1: continue
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            if aspect < ASPECT_MIN: continue
            length = max(rw, rh)
            if length > best_len:
                best_len = length; best = c
        if best is None:  return None, None, None, 0
        M = cv2.moments(best)
        if M["m00"] <= 1e-6: return None, None, None, 0
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        cx_full, cy_full = cx, cy + y0
        return best, cx_full, cy_full, 1

    # Seleção por polaridade (igual ao robot_pedro.py)
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
    if abs(erro) <= DEAD_BAND:
        erro = 0
    return imagem, erro, conf

def calcular_velocidades_auto(erro, base_speed):
    correcao = Kp * float(erro)
    v_esq = base_speed + correcao
    v_dir = base_speed - correcao
    # Permite velocidades mais baixas para correções, mas mantém mínimo razoável
    v_esq = int(np.clip(v_esq, 60, VELOCIDADE_MAX))
    v_dir = int(np.clip(v_dir, 60, VELOCIDADE_MAX))
    return v_esq, v_dir

def enviar_comando_motor_serial(arduino, v_esq, v_dir):
    comando = f"C {v_dir} {v_esq}\n"
    arduino.write(comando.encode('utf-8'))

# ====================== Utilidades ======================
def drive_cap(arduino, v_esq, v_dir, cap=255):
    v_esq=int(np.clip(v_esq, -cap, cap))
    v_dir=int(np.clip(v_dir, -cap, cap))
    enviar_comando_motor_serial(arduino, v_esq, v_dir)

# ====================== Início cego / Pivô (2 fases) / Intersec ======================
def straight_until_seen_then_lost(arduino, camera):
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    saw=False; lost=0; t0=time.time()

    # Começa com velocidade maior para ir mais longe
    initial_speed = int(START_SPEED * 1.15)  # 15% mais rápido (menos que antes)
    drive_cap(arduino, initial_speed, initial_speed); time.sleep(0.1)

    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            img=f.array
            mask=build_binary_mask(img)
            h,w=mask.shape[:2]
            y0=int(h*ROW_BAND_TOP_FRAC); y1=int(h*ROW_BAND_BOTTOM_FRAC)
            band=mask[y0:y1,:]
            band=cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((5,11),np.uint8), iterations=1)
            row_frac = band.sum(axis=1)/(255.0*w)
            present = row_frac.max() >= ROW_PEAK_FRAC_THR

            # Enviar frame para o stream durante a reta inicial
            display_frame = img.copy()
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)
            cv2.putText(display_frame, f"Reta Inicial - Present: {present}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            send_frame_to_stream(display_frame)

            # Mantém velocidade inicial até ver a linha
            current_speed = initial_speed if not saw else START_SPEED
            drive_cap(arduino, current_speed, current_speed)

            if not saw:
                if present: saw=True; lost=0
            else:
                if present: lost=0
                else:
                    lost+=1
                    if lost>=LOSE_FRAMES_START:
                        # Após perder a linha, anda um pouco mais para frente
                        drive_cap(arduino, START_SPEED, START_SPEED); time.sleep(0.15)  # Tempo reduzido
                        drive_cap(arduino,0,0); return True
            if (time.time()-t0)>START_TIMEOUT_S:
                drive_cap(arduino,0,0); return False
            raw.truncate(0); raw.seek(0)
    finally:
        raw.truncate(0)

def spin_in_place_until_seen(arduino, camera, side_hint='L', orient=0):
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    turn_sign = -1 if side_hint=='L' else +1
    # Ajustes de direção de giro por orientação podem ser adicionados aqui se necessário
    # Por enquanto, todas as orientações usam a lógica padrão
    pass
    seen_cnt=0; t0=time.time()
    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            img=f.array
            img_display, err, conf = processar_imagem(img)
            v_esq, v_dir = turn_sign*PIVOT_MIN, -turn_sign*PIVOT_MIN
            drive_cap(arduino, v_esq, v_dir, cap=PIVOT_CAP)

            # Enviar frame para o stream durante o pivot
            mask = build_binary_mask(img_display)
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(img_display, 0.7, mask_color, 0.3, 0)
            cv2.putText(display_frame, f"Pivot - Conf: {conf}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            send_frame_to_stream(display_frame)

            if conf==1:
                seen_cnt += 1
            else:
                seen_cnt = 0

            if seen_cnt >= SEEN_FRAMES:
                drive_cap(arduino, 0, 0)
                return True

            if (time.time()-t0) > PIVOT_TIMEOUT:
                drive_cap(arduino,0,0); return False
            raw.truncate(0); raw.seek(0)
    finally:
        raw.truncate(0)

def forward_align_on_line(arduino, camera):
    """Avança devagar usando P até o erro ficar pequeno por alguns frames."""
    print("   🔄 Iniciando alinhamento na linha...")
    # Pequena pausa para estabilizar após o pivot
    time.sleep(0.3)
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    stable=0; t0=time.time(); lost_frames=0; last_err=0.0; state='FOLLOW'
    frame_count = 0; last_frame_sent = 0

    # Tolerância maior no início (após pivot pode haver instabilidade)
    initial_tolerance_frames = 10
    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            frame_count += 1
            img=f.array
            _, erro, conf = processar_imagem(img)

            # Tolerância maior nos primeiros frames após pivot
            effective_lost_max = LOST_MAX_FRAMES * 2 if frame_count <= initial_tolerance_frames else LOST_MAX_FRAMES

            if conf==1:
                state='FOLLOW'; lost_frames=0; last_err=erro
                v_esq, v_dir = calcular_velocidades_auto(erro, ALIGN_BASE)
                print(f"      Frame {frame_count}: Seguindo | erro={erro:.1f} | vel=({v_esq},{v_dir})")
            else:
                lost_frames+=1
                if lost_frames>=effective_lost_max:
                    state='LOST'
                if state=='LOST':
                    turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                    v_esq, v_dir = int(turn*0.7), int(-turn*0.7)  # giro suave
                    print(f"      Frame {frame_count}: Perdido! Procurando | vel=({v_esq},{v_dir})")
                else:
                    v_esq, v_dir = ALIGN_BASE, ALIGN_BASE
                    print(f"      Frame {frame_count}: Sem linha | vel=reto (tolerancia: {effective_lost_max})")

            drive_cap(arduino, v_esq, v_dir, cap=ALIGN_CAP)

            # Criar frame para visualização
            display_frame = img.copy()
            mask = build_binary_mask(display_frame)
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

            # HUD de alinhamento
            cv2.putText(display_frame, f"Alinhamento - Frame {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Estado: {state}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(display_frame, f"Confiança: {conf}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(display_frame, f"Erro: {erro:.1f}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)
            cv2.putText(display_frame, f"Estável: {stable}/{ALIGN_STABLE}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            # Enviar frame a cada 5 frames ou quando importante
            if frame_count - last_frame_sent >= 5 or stable >= ALIGN_STABLE:
                send_frame_to_stream(display_frame)
                last_frame_sent = frame_count

            if conf==1 and abs(erro)<=ALIGN_TOL_PIX:
                stable+=1
                print(f"      → Estável: {stable}/{ALIGN_STABLE} frames")
            else:
                stable=0

            if stable>=ALIGN_STABLE:
                print(f"      ✅ Alinhamento concluído após {frame_count} frames!")
                # Frame final de sucesso
                cv2.putText(display_frame, "ALINHAMENTO CONCLUIDO!", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                send_frame_to_stream(display_frame)

                drive_cap(arduino, 70, 70, cap=ALIGN_CAP); time.sleep(0.10)
                drive_cap(arduino, 0, 0); return True

            elapsed = time.time()-t0
            if elapsed > ALIGN_TIMEOUT:
                print(f"      ❌ Timeout após {elapsed:.1f}s ({frame_count} frames)")
                drive_cap(arduino,0,0); return False

            raw.truncate(0); raw.seek(0)
    finally:
        raw.truncate(0)

def best_intersection_in_band(pts, h, band_y0, band_y1):
    """Escolhe a melhor interseção: primeiro tenta na banda, senão aceita fora da banda"""
    # Primeiro tenta encontrar na banda principal
    cand_in_band = None
    best_y_in_band = -1

    cand_out_band = None
    best_y_out_band = h  # Começa com o maior y possível (mais longe)

    for (x,y) in pts:
        if band_y0 <= y <= band_y1:
            # Dentro da banda - escolhe a mais próxima (maior y)
            if y > best_y_in_band:
                best_y_in_band = y
                cand_in_band = (x,y)
        else:
            # Fora da banda - escolhe a mais próxima (menor y)
            if y < best_y_out_band:
                best_y_out_band = y
                cand_out_band = (x,y)

    # Prioriza interseções dentro da banda, mas aceita fora se necessário
    return cand_in_band if cand_in_band is not None else cand_out_band

def go_to_next_intersection(arduino, camera):
    """
    Vai até a próxima interseção usando a lógica robusta do robot_pedro.py
    """
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    last_err = 0.0
    lost_frames = 0
    # Estados: 'FOLLOW', 'LOST', 'APPROACHING', 'STOPPING', 'STOPPED'
    state = 'FOLLOW'
    action_start_time = 0.0
    last_known_y = -1.0  # Última posição Y válida da interseção

    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            img = f.array
            mask = build_binary_mask(img)
            img, erro, conf = processar_imagem(img)

            # Debug: verificar se está detectando linha
            if conf == 0:
                print(f"   ⚠️  Linha perdida! erro={erro}, conf={conf}")
            else:
                print(f"   ✅ Linha OK: erro={erro:.1f}, conf={conf}")

            # Encontrar a interseção alvo (filtrar bordas, focar no centro)
            intersections, detected_lines = detect_intersections(mask)
            target_intersection = None
            target_y = -1

            if intersections:
                h, w = img.shape[:2]
                # Filtrar interseções muito próximas das bordas (provavelmente falsas)
                border_margin = int(w * 0.15)  # 15% das bordas
                filtered_intersections = [
                    inter for inter in intersections
                    if border_margin <= inter[0] <= w - border_margin
                ]

                if filtered_intersections:
                    filtered_intersections.sort(key=lambda p: p[1], reverse=True)  # Ordena por Y decrescente
                    target_intersection = filtered_intersections[0]
                    target_y = target_intersection[1]
                    print(f"   📍 Target intersection: {target_intersection} (x={target_intersection[0]}, y={target_y}) - FILTRADA")
                else:
                    # Fallback: usar interseção mais próxima do centro se nenhuma no centro
                    center_x = w // 2
                    intersections.sort(key=lambda p: (abs(p[0] - center_x), -p[1]))  # Centro X, depois Y alto
                    target_intersection = intersections[0]
                    target_y = target_intersection[1]
                    print(f"   📍 Target intersection: {target_intersection} (x={target_intersection[0]}, y={target_y}) - FALLBACK")

            h, w = img.shape[:2]
            Y_START_SLOWING = h * Y_START_SLOWING_FRAC
            Y_TARGET_STOP = h * Y_TARGET_STOP_FRAC

            # Debug: mostrar valores importantes
            if target_y != -1:
                print(f"   🎯 Interseção Y={target_y:.0f}, Y_TARGET_STOP={Y_TARGET_STOP:.0f}, State={state}")
                # Verificar se deve entrar em APPROACHING
                should_approach = target_y > Y_START_SLOWING
                print(f"   🔍 Should approach: {should_approach} (Y > {Y_START_SLOWING:.0f})")

            # --- Máquina de Estados de Controle (do robot_pedro.py) ---

            # 1. Transições de Estado
            if state == 'FOLLOW':
                print(f"   🔄 State machine: conf={conf}, target_y={target_y}, Y_START_SLOWING={Y_START_SLOWING:.0f}")
                # Verificar se devemos aproximar (independente de conf atual)
                if target_y > Y_START_SLOWING and target_y != -1:
                    print(f"   🎯 Interseção detectada em Y={target_y:.0f}! Iniciando aproximação (Y_START_SLOWING={Y_START_SLOWING:.0f})")
                    state = 'APPROACHING'
                    last_known_y = target_y
                    lost_frames = 0
                elif conf == 0:
                    lost_frames += 1
                    if lost_frames >= LOST_MAX_FRAMES:
                        state = 'LOST'
                        last_known_y = -1.0
                else:
                    lost_frames = 0
                    print(f"   ⏳ Aguardando aproximação: target_y={target_y:.0f} <= Y_START_SLOWING={Y_START_SLOWING:.0f}")
                    last_err = erro
                    last_known_y = -1.0

            elif state == 'APPROACHING':
                # Verifica a perda de linha, mas com tolerância
                if conf == 0:
                    lost_frames += 1
                    print(f"   ⚠️  Aproximando, confiança perdida! (Frame {lost_frames})")

                    if lost_frames >= LOST_MAX_FRAMES:
                        print("   ❌ Linha perdida durante aproximação.")
                        state = 'LOST'
                        last_known_y = -1.0

                else:
                    lost_frames = 0

                    # Atualiza a posição conhecida da interseção
                    if target_y != -1:
                         last_known_y = target_y

                         # GATILHO 1: Atingimos o alvo de Y?
                         if last_known_y >= Y_TARGET_STOP:
                            print("   🛑 Alvo (Y_TARGET_STOP) atingido. 'Andando mais um pouco'...")
                            state = 'STOPPING'
                            action_start_time = time.time()
                            last_known_y = -1.0 # Reseta para a próxima

                    # GATILHO 2: Interseção desapareceu completamente (backup)
                    if target_y == -1 and last_known_y > Y_START_SLOWING:
                        print(f"   🛑 Interseção desapareceu (era Y={last_known_y:.0f}). Parando...")
                        state = 'STOPPING'
                        action_start_time = time.time()
                        last_known_y = -1.0 # Reseta para a próxima

            elif state == 'STOPPING':
                if (time.time() - action_start_time) > CRAWL_DURATION_S:
                    print("   ✅ Parada completa na interseção!")
                    state = 'STOPPED'

            elif state == 'LOST':
                if conf == 1:
                    print("   ✅ Linha reencontrada.")
                    state = 'FOLLOW'
                    lost_frames = 0
                    last_err = erro
                    last_known_y = -1.0

            # 2. Ações de Estado (Definir velocidades)
            if state == 'FOLLOW':
                if conf == 1:
                    # Mantém velocidade base constante, apenas corrige com P
                    v_esq, v_dir = calcular_velocidades_auto(erro, VELOCIDADE_BASE)
                else:
                    # Janela de tolerância: continua reto
                    v_esq, v_dir = VELOCIDADE_BASE, VELOCIDADE_BASE

            elif state == 'APPROACHING':
                if conf == 0:
                    # Continua reto em velocidade reduzida
                    base_speed = int(np.clip(VELOCIDADE_BASE * 0.35, V_MIN, VELOCIDADE_MAX))
                    v_esq, v_dir = calcular_velocidades_auto(0, base_speed)
                else:
                    # Frenagem gradual baseada em last_known_y
                    progress = 0.0
                    if (Y_TARGET_STOP - Y_START_SLOWING) > 0:
                        progress = (last_known_y - Y_START_SLOWING) / (Y_TARGET_STOP - Y_START_SLOWING)

                    speed_factor = 1.0 - np.clip(progress, 0.0, 1.0)
                    current_base_speed = (VELOCIDADE_BASE - CRAWL_SPEED) * speed_factor + CRAWL_SPEED
                    base_speed = int(np.clip(current_base_speed, CRAWL_SPEED, VELOCIDADE_MAX))
                    v_esq, v_dir = calcular_velocidades_auto(erro, base_speed)

            elif state == 'STOPPING':
                # "Anda mais um pouco" - crawl reto
                v_esq, v_dir = CRAWL_SPEED, CRAWL_SPEED

            elif state == 'STOPPED':
                v_esq, v_dir = 0, 0

            elif state == 'LOST':
                # Lógica de busca
                turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                v_esq, v_dir = int(turn), int(-turn)

            drive_cap(arduino, v_esq, v_dir, cap=ALIGN_CAP)

            # ---------------- VISUALIZAÇÃO ----------------
            display_frame = img.copy()
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

            # desenha linhas (verde)
            for rho, theta in detected_lines:
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

            # Destaque a interseção alvo
            if target_intersection is not None:
                cv2.circle(display_frame, target_intersection, 15, (255, 0, 255), 3)

            # HUD com estado atual
            state_color = (0, 255, 0)  # Verde para FOLLOW
            if state == 'LOST': state_color = (0, 0, 255)  # Vermelho
            elif state == 'APPROACHING': state_color = (0, 255, 255)  # Amarelo
            elif state == 'STOPPING': state_color = (255, 0, 255)  # Magenta
            elif state == 'STOPPED': state_color = (255, 0, 0)  # Azul

            cv2.putText(display_frame, f"State: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            cv2.putText(display_frame, f"Conf: {conf}  Lost: {lost_frames}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)
            cv2.putText(display_frame, f"Y_target: {target_y:.0f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            send_frame_to_stream(display_frame)

            if state == 'STOPPED':
                return True

            raw.truncate(0); raw.seek(0)

            # Timeout de segurança
            if (time.time() - time.time()) > 20.0:
                print("   ❌ Timeout na detecção de interseção")
                drive_cap(arduino, 0, 0)
                return False

    finally:
        raw.truncate(0)

# ====================== Planejamento e Execução ======================
def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def front_left_right_corners(sx,sy,orient):
    # Retorna (left_corner, right_corner) relativos à direção de movimento
    # left_corner: interseção alcançável virando para esquerda
    # right_corner: interseção alcançável virando para direita
    # Interseções acessíveis baseadas no quadrado (sx,sy)
    # Para quadrado X,Y: interseções são (X,Y), (X,Y+1), (X+1,Y), (X+1,Y+1)
    # Mas acessíveis dependem da orientação
    if orient==0:  return ( (sx,sy),     (sx,sy+1) )     # Norte: (X,Y), (X,Y+1)
    if orient==1:  return ( (sx+1,sy),   (sx+1,sy+1) )   # Leste: (X+1,Y), (X+1,Y+1)
    if orient==2:  return ( (sx+1,sy+1), (sx+1,sy) )   # Sul: (X+1,Y+1), (X+1,Y)
    if orient==3:  return ( (sx+1,sy),   (sx,sy) )     # Oeste: (X+1,Y), (X,Y)
    raise ValueError

def get_accessible_intersections(sx, sy, orient):
    """Retorna todas as interseções acessíveis de um quadrado em uma orientação"""
    left_corner, right_corner = front_left_right_corners(sx, sy, orient)
    return [left_corner, right_corner]

def find_best_accessible_intersection(path, cur_dir):
    """
    Encontra a interseção no path A* que seja acessível da orientação atual,
    escolhendo a mais próxima do início do caminho.
    """
    if len(path) <= 1:
        return path[0] if path else None

    # Para qualquer quadrado, determinar interseções acessíveis
    # Assumindo que estamos saindo do quadrado path[0]
    start_square = path[0]
    sx, sy = start_square

    accessible = get_accessible_intersections(sx, sy, cur_dir)

    # Procurar a interseção no path que seja acessível e mais próxima do início
    for intersection in path[1:]:  # Começar do path[1] (primeira interseção)
        if intersection in accessible:
            return intersection

    # Se nenhuma interseção do path for acessível, escolher a acessível com menor
    # distância para a primeira interseção do path
    target_intersection = path[1]
    best_accessible = min(accessible,
                         key=lambda inter: manhattan(inter, target_intersection))

    return best_accessible

def a_star(start,goal,grid=(5,5)):
    open_set={start}; came={}; g={start:0}; f={start:manhattan(start,goal)}
    W,H=grid
    def neigh(n):
        x,y=n
        cand=[]
        if y-1>=0: cand.append((x,y-1))
        if x+1<W : cand.append((x+1,y))
        if y+1<H: cand.append((x,y+1))
        if x-1>=0: cand.append((x-1,y))
        return cand
    while open_set:
        cur=min(open_set,key=lambda n:f.get(n,1e18))
        if cur==goal:
            path=[cur]
            while cur in came: cur=came[cur]; path.append(cur)
            return list(reversed(path))
        open_set.remove(cur)
        for nxt in neigh(cur):
            ng=g[cur]+1
            if ng<g.get(nxt,1e18):
                came[nxt]=cur; g[nxt]=ng; f[nxt]=ng+manhattan(nxt,goal)
                open_set.add(nxt)
    return None

def orientation_of_step(a,b):
    # Coordenadas (linha, coluna) - linha cresce para baixo, coluna cresce para direita
    # Norte: linha diminui (b[0] < a[0])
    # Sul: linha aumenta (b[0] > a[0])
    # Leste: coluna aumenta (b[1] > a[1])
    # Oeste: coluna diminui (b[1] < a[1])
    if b[0] < a[0]: return 0  # Norte
    if b[0] > a[0]: return 2  # Sul
    if b[1] > a[1]: return 1  # Leste
    return 3  # Oeste
def relative_turn(cur_dir,want_dir): return {0:'F',1:'R',2:'U',3:'L'}[(want_dir-cur_dir)%4]

def dir_name(d):
    return {0:'Norte', 1:'Leste', 2:'Sul', 3:'Oeste'}[d]

def leave_square_to_best_corner(arduino, camera, sx, sy, cur_dir, target, target_intersection=None, return_arrival_dir=True):
    """
    Sai do quadrado usando a orientação declarada (assumida correta).
    path: caminho A* completo para usar interseção específica se disponível
    """
    print(f"🚶 Saindo do quadrado ({sx},{sy})")
    print(f"   Orientação: {'Norte' if cur_dir == 0 else 'Leste' if cur_dir == 1 else 'Sul' if cur_dir == 2 else 'Oeste'}")
    print(f"   Destino: {target}")

    left_corner, right_corner = front_left_right_corners(sx, sy, cur_dir)

    # Se temos uma interseção alvo específica do A*, usar ela diretamente se possível
    if target_intersection is not None:
        if target_intersection == left_corner:
            side_hint = 'L'
            chosen = left_corner
        elif target_intersection == right_corner:
            side_hint = 'R'
            chosen = right_corner
        else:
            # Interseção alvo não acessível diretamente, usar fallback baseado no target final
            dl = manhattan(left_corner, target)
            dr = manhattan(right_corner, target)
            side_hint = 'L' if dl <= dr else 'R'
            chosen = left_corner if side_hint=='L' else right_corner
            print(f"   ⚠️ Interseção alvo {target_intersection} não acessível, usando fallback")
    else:
        # Fallback para lógica antiga
        dl = manhattan(left_corner, target)
        dr = manhattan(right_corner, target)
        side_hint = 'L' if dl <= dr else 'R'
        chosen = left_corner if side_hint=='L' else right_corner

    turn_desc = {'L':'esquerda', 'R':'direita'}[side_hint]
    print(f"   Escolhendo canto {chosen} (virada: {turn_desc})")
    if target_intersection is None:
        print(f"   Distância Manhattan: {dl} vs {dr}")

    # Reta cega
    if not straight_until_seen_then_lost(arduino, camera):
        print("✗ Falha na reta inicial.")
        return None, None, False

    # Pivô: girar até ver a linha
    if not spin_in_place_until_seen(arduino, camera, side_hint=side_hint, orient=cur_dir):
        print("✗ Falha no pivô (não viu linha).")
        return None, None, False

    # Alinhamento rápido: centralizar a linha por alguns frames
    print("   🎯 Alinhando linha no centro...")
    raw_temp = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    align_count = 0
    try:
        for f in camera.capture_continuous(raw_temp, format="bgr", use_video_port=True):
            if align_count >= 10:  # Máximo 10 frames de alinhamento
                break
            img_temp = f.array
            _, erro, conf = processar_imagem(img_temp)

            if conf == 1 and abs(erro) <= 10:  # Já está bem centralizado
                print(f"   ✅ Linha alinhada (erro: {erro:.1f})")
                break

            # Movimento de correção simples
            if conf == 1:
                base_speed = 70
                if erro > 5:  # Linha à direita, virar esquerda
                    drive_cap(arduino, base_speed-20, base_speed+20, cap=ALIGN_CAP)
                elif erro < -5:  # Linha à esquerda, virar direita
                    drive_cap(arduino, base_speed+20, base_speed-20, cap=ALIGN_CAP)
                else:  # Centralizado
                    drive_cap(arduino, base_speed, base_speed, cap=ALIGN_CAP)
            else:
                drive_cap(arduino, 60, 60, cap=ALIGN_CAP)  # Anda devagar se perdeu linha

            time.sleep(0.05)  # Frame rate control
            align_count += 1
            raw_temp.truncate(0)
    finally:
        drive_cap(arduino, 0, 0); time.sleep(0.1)  # Para antes de continuar
        raw_temp.truncate(0)

    # Segue para 1ª intersecção
    if not go_to_next_intersection(arduino, camera):
        print("✗ Falha ao alcançar a intersecção.")
        return None, None, False

    # O pivot virou o robô para a direção do corner escolhido
    # Atualizar cur_dir baseado no side_hint
    if side_hint == 'L':
        cur_dir = (cur_dir - 1) % 4  # Virou para esquerda
    elif side_hint == 'R':
        cur_dir = (cur_dir + 1) % 4  # Virou para direita

    print(f"✅ Pivot concluído - agora virado para {dir_name(cur_dir)}")

    if return_arrival_dir:
        # O robô chega na interseção vindo da direção atual
        arrival_dir = cur_dir
        print(f"📍 Chegando na interseção {chosen} vindo do {dir_name(arrival_dir)}")
        return chosen, cur_dir, True, arrival_dir
    else:
        return chosen, cur_dir, True

# exec_turn removida - ações executadas diretamente em follow_path usando lógica do robot_pedro.py

def follow_path(arduino, start_node, start_dir, path, camera, arrival_dir=None):
    """
    O robô JÁ ESTÁ na primeira interseção (start_node) após leave_square_to_best_corner.
    arrival_dir: direção de chegada na primeira interseção (0=N, 1=L, 2=S, 3=W)
    Se None, assume que arrival_dir = start_dir
    """
    cur_node=start_node
    # Se não especificada, assume que chegou virado para start_dir
    actual_arrival_dir = arrival_dir if arrival_dir is not None else start_dir
    cur_dir = actual_arrival_dir  # Começa com a direção de chegada

    print(f"🚶🏁 Chegando na primeira interseção {start_node} vindo do {dir_name(actual_arrival_dir)}")

    drive_cap(arduino,0,0); time.sleep(0.1)

    # Mostra o caminho completo
    print("🗺️  CAMINHO CALCULADO PELO A*:")
    path_str = " → ".join([f"({x},{y})" for x,y in path])
    print(f"   {path_str}")
    print(f"   Total: {len(path)-1} movimentos")
    print()

    # Verifica se já estamos no destino
    if start_node == path[-1]:
        print(f"🎯 Já estamos no destino ({start_node[0]},{start_node[1]})!")
        return cur_node,cur_dir,True

    # O robô JÁ ESTÁ na primeira interseção após leave_square_to_best_corner
    # Não precisamos ir para lugar nenhum, apenas confirmar que estamos na posição certa
    print(f"🚶🏁 Já estamos na primeira interseção {start_node}")
    cur_node = start_node
    print()

    # Executa cada passo do caminho (começando do segundo nó)
    for i in range(1,len(path)):
        nxt=path[i]
        want=orientation_of_step(cur_node, nxt)
        rel=relative_turn(cur_dir,want)

        # Debug: mostra cálculos
        print(f"   DEBUG: cur_node={cur_node}, nxt={nxt}, cur_dir={cur_dir}({dir_name(cur_dir)}), want={want}({dir_name(want)}), rel={rel}")

        # Mostra cada virada específica
        turn_names = {'F':'reto', 'L':'esquerda', 'R':'direita', 'U':'meia-volta'}
        print(f"🔄 Intersecção ({cur_node[0]},{cur_node[1]}): virar {turn_names[rel]} para ({nxt[0]},{nxt[1]})")
        print(f"   📍 cur_dir={cur_dir}, want={want}, rel={rel}")

        # ⚠️  IMPORTANTE: Para completamente antes de virar
        drive_cap(arduino, 0, 0); time.sleep(0.3)
        print(f"   🛑 Parado para executar giro")

        # Executa a ação baseada no giro relativo (lógica do robot_pedro.py)
        if rel == 'F':
            # GO_STRAIGHT: Já está virado para a direção certa, apenas atualiza direção
            print("   ➡️  Já virado para a direção certa, seguindo em frente...")
            cur_dir = want

        elif rel == 'L':
            # TURN_LEFT: Virar 90° esquerda
            print(f"   ↪️  Virando esquerda: drive_cap({arduino}, {-TURN_SPEED}, {TURN_SPEED}) por {TURN_DURATION_S}s")
            drive_cap(arduino, -TURN_SPEED, TURN_SPEED)
            time.sleep(TURN_DURATION_S)
            print(f"   🛑 Parando giro esquerdo...")
            drive_cap(arduino, 0, 0); time.sleep(0.3)
            print("   ✅ Virou esquerda")
            cur_dir = want

        elif rel == 'R':
            # TURN_RIGHT: Virar 90° direita
            print(f"   ↩️  Virando direita: drive_cap({arduino}, {TURN_SPEED}, {-TURN_SPEED}) por {TURN_DURATION_S}s")
            drive_cap(arduino, TURN_SPEED, -TURN_SPEED)
            time.sleep(TURN_DURATION_S)
            print(f"   🛑 Parando giro direito...")
            drive_cap(arduino, 0, 0); time.sleep(0.3)
            print("   ✅ Virou direita")
            cur_dir = want

        elif rel == 'U':
            # U-turn: Meia-volta (180°)
            print("   🔄 Fazendo meia-volta...")
            drive_cap(arduino, TURN_SPEED, -TURN_SPEED)
            time.sleep(1.3)  # U-turn ajustado para 1.1s conforme solicitado
            drive_cap(arduino, 0, 0); time.sleep(0.4)
            print("   ✅ Meia-volta completa")
            cur_dir = want

        print(f"   ✅ Ação executada")

        # Agora vai para a próxima interseção seguindo a linha
        if not go_to_next_intersection(arduino, camera):
            print(f"   ❌ Falha ao alcançar ({nxt[0]},{nxt[1]})")
            return cur_node,cur_dir,False

        # Após o movimento, o robô mantém a direção 'want' para a qual estava indo
        print(f"   ✅ Chegou em ({nxt[0]},{nxt[1]})")
        print()
        cur_node=nxt
        cur_dir = want  # Mantém a direção para a qual estava indo

    print(f"🎯 Chegou ao destino final!")
    return cur_node,cur_dir,True

# =================================== CONTROLE REMOTO ===================================
def init_remote_control():
    """Inicializa conexão com o servidor de controle remoto"""
    try:
        context = zmq.Context()
        req_socket = context.socket(zmq.REQ)
        req_socket.connect(f"tcp://{SERVER_IP}:5005")
        print(f"🕹️  Controle remoto conectado ao servidor {SERVER_IP}")
        return req_socket
    except Exception as e:
        print(f"⚠️  Erro ao conectar controle remoto: {e}")
        return None

def get_remote_key(req_socket):
    """Obtém tecla do controle remoto"""
    if req_socket is None:
        return None
    try:
        msg = {"from": "robot", "cmd": "key_request"}
        req_socket.send_pyobj(msg)
        reply = req_socket.recv_pyobj()
        key = reply.get("key", "")
        return key if key else None
    except Exception as e:
        print(f"⚠️  Erro ao obter tecla remota: {e}")
        return None

def manual_control(arduino, key):
    """Controle manual baseado na tecla pressionada"""
    if key == 'w':  # Frente
        drive_cap(arduino, VELOCIDADE_BASE, VELOCIDADE_BASE)
        print("🕹️  MANUAL: Frente")
    elif key == 's':  # Trás
        drive_cap(arduino, -VELOCIDADE_BASE, -VELOCIDADE_BASE)
        print("🕹️  MANUAL: Trás")
    elif key == 'a':  # Esquerda
        drive_cap(arduino, -VELOCIDADE_BASE, VELOCIDADE_BASE)
        print("🕹️  MANUAL: Girando esquerda")
    elif key == 'd':  # Direita
        drive_cap(arduino, VELOCIDADE_BASE, -VELOCIDADE_BASE)
        print("🕹️  MANUAL: Girando direita")
    elif key == 'stop':  # Parar
        drive_cap(arduino, 0, 0)
        print("🕹️  MANUAL: Parado")
    elif key == 'm':  # Toggle modo
        print("🕹️  Toggle modo manual/automatico")

# =================================== MAIN ===================================
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--square', type=int, nargs=2, required=True, metavar=('SX','SY'))
    ap.add_argument('--orient', type=str, required=True, help='N/E/S/W')
    ap.add_argument('--target', type=int, nargs=2, required=True, metavar=('TX','TY'))
    ap.add_argument('--no-return', action='store_true')
    return ap.parse_args()

# Variáveis globais para streaming
stream_socket = None
stream_context = None

# Controle de detecção inicial
initial_frames_ignored = 0
IGNORE_INITIAL_FRAMES = 15  # Ignora primeiras 15 frames para evitar detecção de chão

def init_streaming():
    """Inicializa o socket ZMQ para streaming"""
    global stream_socket, stream_context
    if stream_socket is None:
        stream_context = zmq.Context()
        stream_socket = stream_context.socket(zmq.PUB)
        stream_socket.bind('tcp://*:5555')
        print("📹 Streaming ZMQ inicializado em tcp://*:5555")

def send_frame_to_stream(display_frame):
    """Envia um frame específico para o stream ZMQ"""
    global stream_socket
    if stream_socket:
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        stream_socket.send(base64.b64encode(buffer))

def send_basic_frame(camera, message="Processando..."):
    """Envia um frame básico da câmera com uma mensagem"""
    try:
        raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
        camera.capture(raw, format="bgr", use_video_port=True)
        img = raw.array
        mask = build_binary_mask(img)

        # Visualização básica
        display_frame = img.copy()
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

        # HUD básico
        cv2.putText(display_frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        send_frame_to_stream(display_frame)
    except Exception as e:
        print(f"Erro ao enviar frame básico: {e}")

def main():
    args=parse_args()
    sx,sy=args.square; tx,ty=args.target
    orient=args.orient.strip().upper()
    if orient not in ('N','E','S','W','O'): raise SystemExit("orient deve ser N/E/S/W")
    cur_dir={'N':0,'E':1,'S':2,'W':3,'O':3}[orient]
    if not (0<=sx<=3 and 0<=sy<=3): raise SystemExit("square 0..3 0..3")
    if not (0<=tx<=4 and 0<=ty<=4): raise SystemExit("target 0..4 0..4")
    target=(tx,ty)

    # Inicializar controle remoto
    remote_socket = init_remote_control()

    # Inicializar câmera e streaming
    camera = PiCamera(); camera.resolution=(IMG_WIDTH, IMG_HEIGHT); camera.framerate=24
    time.sleep(1.0)  # warm-up mais longo
    init_streaming()  # Inicializar ZMQ

    # Aguardar um pouco mais antes do primeiro frame
    time.sleep(0.5)

    # Enviar primeiro frame básico com retry
    for attempt in range(3):
        try:
            send_basic_frame(camera, "Sistema inicializado - aguardando comando")
            print("📹 Primeiro frame enviado com sucesso")
            break
        except Exception as e:
            print(f"Tentativa {attempt+1} falhou: {e}")
            time.sleep(0.5)
    else:
        print("⚠️ Não conseguiu enviar primeiro frame, continuando...")

    arduino = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=1); time.sleep(2)
    try:
        arduino.write(b'A10\n')
        try: print("Arduino:", arduino.readline().decode('utf-8').strip())
        except Exception: pass
    except Exception: pass

    # Estados do robô
    manual_mode = False
    last_key = None
    auto_state = "INIT"  # Estados: INIT, LEAVING, NAVIGATING, RETURNING, DONE

    try:
        print(f"🏠 INÍCIO: Quadrado ({sx},{sy})")
        print(f"📦 DESTINO: Nó ({tx},{ty})")
        print("🤖 MODO AUTOMÁTICO")
        print()

        # Determinar interseção inicial baseada na orientação
        accessible = get_accessible_intersections(sx, sy, cur_dir)
        start_intersection = min(accessible, key=lambda inter: manhattan(inter, (tx, ty)))
        print(f"🎯 Interseção inicial escolhida: {start_intersection} (baseado na orientação e destino)")

        # Calcular A* da interseção inicial para o destino
        print("🤖 EXECUTANDO A* PARA CALCULAR CAMINHO...")
        send_basic_frame(camera, "Calculando caminho A*...")

        path = a_star(start_intersection, (tx, ty), GRID_NODES)
        if path is None:
            print("❌ Nenhum caminho encontrado pelo A*.")
            send_basic_frame(camera, "ERRO: Caminho nao encontrado!")
            return

        print(f"🗺️ CAMINHO: {' -> '.join([f'({x},{y})' for x,y in path])}")
        send_basic_frame(camera, f"Caminho: {' -> '.join([f'({x},{y})' for x,y in path])}")

        # Determinar a melhor interseção inicial baseada na orientação
        target_intersection = find_best_accessible_intersection(path, cur_dir)
        print(f"🎯 Melhor interseção acessível: {target_intersection} (baseado na orientação)")

        # Variáveis para o modo automático
        start_node = None
        back_path = None
        arrival_dir = None

        while True:  # Loop principal com controle remoto
            # Verificar comandos remotos
            remote_key = get_remote_key(remote_socket)

            if remote_key == 'm':
                manual_mode = not manual_mode
                if manual_mode:
                    print("🕹️  MODO MANUAL ATIVADO - Use W/A/S/D para controlar")
                    send_basic_frame(camera, "MODO MANUAL - Use W/A/S/D")
                    drive_cap(arduino, 0, 0)  # Parar antes de mudar modo
                    last_key = None
                else:
                    print("🤖 MODO AUTOMÁTICO ATIVADO")
                    send_basic_frame(camera, "MODO AUTOMATICO")
                    drive_cap(arduino, 0, 0)  # Parar antes de mudar modo
                    last_key = None
                    auto_state = "INIT"  # Resetar estado automático
                time.sleep(0.5)  # Debounce
                continue

            if manual_mode:
                # Modo manual: responder diretamente aos comandos
                if remote_key in ['w', 'a', 's', 'd']:
                    manual_control(arduino, remote_key)
                    last_key = remote_key
                elif remote_key == 'stop' or (last_key and not remote_key):
                    manual_control(arduino, 'stop')
                    last_key = None
                # Se não há tecla, continua o último comando (para manter movimento)

            else:
                # Modo automático: executar lógica de navegação
                if auto_state == "INIT":
                    send_basic_frame(camera, f"Quadrado ({sx},{sy}) -> No ({tx},{ty})")
                    auto_state = "LEAVING"

                elif auto_state == "LEAVING":
                    print("🚶 Executando leave_square_to_best_corner...")
                    result = leave_square_to_best_corner(arduino, camera, sx, sy, cur_dir, target, target_intersection)
                    print(f"✅ leave_square_to_best_corner retornou: {result}")
                    if len(result) == 4:
                        start_node, cur_dir, ok, arrival_dir = result
                    else:
                        start_node, cur_dir, ok = result
                        arrival_dir = cur_dir  # fallback
                    if not ok:
                        print("❌ Falha na saída.")
                        send_basic_frame(camera, "ERRO: Falha na saida")
                        return

                    print("🔄 Mudando para NAVIGATING")
                    auto_state = "NAVIGATING"

                elif auto_state == "NAVIGATING":
                    # Recalcular A* da interseção escolhida para o destino
                    print(f"🔄 Recalculando A* da interseção {start_node} para destino {target}")
                    optimized_path = a_star(start_node, target, GRID_NODES)
                    if optimized_path is None:
                        print("❌ Nenhum caminho encontrado da interseção escolhida.")
                        send_basic_frame(camera, "ERRO: Caminho nao encontrado!")
                        return

                    print(f"🗺️ CAMINHO OTIMIZADO: {' -> '.join([f'({x},{y})' for x,y in optimized_path])}")
                    send_basic_frame(camera, f"Navegando: {' -> '.join([f'({x},{y})' for x,y in optimized_path])}")

                    _, cur_dir, ok = follow_path(arduino, start_node, cur_dir, optimized_path, camera, arrival_dir)
                    if not ok:
                        print("❌ Falha na navegação.")
                        send_basic_frame(camera, "ERRO: Falha na navegacao")
                        return
                    print("✅ Entrega realizada com sucesso!")
                    send_basic_frame(camera, "Entrega realizada!")
                    auto_state = "RETURNING" if not args.no_return else "DONE"

                elif auto_state == "RETURNING":
                    print("🔄 CALCULANDO CAMINHO DE RETORNO...")
                    send_basic_frame(camera, "Calculando retorno...")

                    back_path = a_star(target, (sx, sy), GRID_NODES)
                    if back_path is None:
                        print("❌ Nenhum caminho de retorno encontrado.")
                        send_basic_frame(camera, "ERRO: Caminho retorno nao encontrado")
                        return

                    print(f"🔙 CAMINHO RETORNO: {' -> '.join([f'({x},{y})' for x,y in back_path])}")
                    send_basic_frame(camera, f"Retorno: {' -> '.join([f'({x},{y})' for x,y in back_path])}")

                    # Para o retorno, assumimos que chegamos virados para cur_dir
                    _, _, ok = follow_path(arduino, target, cur_dir, back_path, camera, cur_dir)
                    if ok:
                        print("✅ Retornou ao ponto inicial!")
                        send_basic_frame(camera, "Retorno concluido!")
                    else:
                        print("❌ Falhou no retorno.")
                        send_basic_frame(camera, "ERRO: Falha no retorno")
                    auto_state = "DONE"

                elif auto_state == "DONE":
                    print("🎯 Missão completa!")
                    send_basic_frame(camera, "MISSAO COMPLETA")
                    break

            time.sleep(0.1)  # Pequena pausa para não sobrecarregar

    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        try:
            enviar_comando_motor_serial(arduino, 0, 0)
            arduino.write(b'a\n'); arduino.close()
        except Exception: pass
        camera.close()
        if remote_socket:
            remote_socket.close()
        return

    finally:
        try:
            enviar_comando_motor_serial(arduino, 0, 0)
            arduino.write(b'a\n'); arduino.close()
        except Exception: pass
        camera.close()
        if remote_socket:
            remote_socket.close()

if __name__=='__main__':
    main()
