# delivery_delivery_v12.py — Delivery 4x4 (robot2 vision/control inline)
# Novidades v12:
#  • Pós-pivô em 2 fases: (a) girar até VER a linha; (b) avançar devagar usando P até CENTRALIZAR.
#    Isso resolve o “não anda o suficiente depois do pivô” e melhora o lock.
#  • Intersecções menos “estritas”: banda mais baixa e estabilidade em 2 frames.
#  • Mantém todas as rotinas de visão/controle idênticas ao robot2.py.
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2, time, numpy as np, serial, argparse, zmq, base64

# ============================= PARÂMETROS (iguais ao robot2) =============================
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 200  # Aumentado de 180 para reduzir falsos positivos
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
Kp = 0.75             # Ganho do controlador P
VELOCIDADE_MAX = 255
E_MAX_PIX       = IMG_WIDTH // 2
V_MIN           = 0
SEARCH_SPEED    = 120
LOST_MAX_FRAMES = 5
DEAD_BAND       = 6
ROI_BOTTOM_FRAC = 0.55
MIN_AREA_FRAC   = 0.008  # Aumentado de 0.004 para reduzir falsos positivos
MAX_AREA_FRAC   = 0.25
ASPECT_MIN      = 3.0    # Aumentado de 2.0 para linhas mais alongadas
LINE_POLARITY   = 'auto'
USE_ADAPTIVE    = False

PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200

# ======== DELIVERY (extra) ========
GRID_NODES = (5, 5)       # 4x4 quadrados → 5x5 nós
START_SPEED  = 100        # reta cega
TURN_SPEED   = 150        # giros 90/180 (crus)

# PIVÔ e aquisição pós-pivô
PIVOT_CAP       = 150     # limite superior do pivô
PIVOT_MIN       = 120     # mínimo para vencer atrito
PIVOT_TIMEOUT   = 7.0
SEEN_FRAMES     = 2       # frames consecutivos "vendo" a linha para sair do giro
ALIGN_BASE      = 90      # velocidade base na fase de alinhamento (P)  [aumentada para mover o carrinho]
ALIGN_CAP       = 120     # cap de segurança na fase de alinhamento [reduzido]
ALIGN_TOL_PIX   = 8       # centralização final
ALIGN_STABLE    = 3       # frames estáveis [reduzido para ser menos rigoroso]
ALIGN_TIMEOUT   = 6.0     # tempo máx. alinhando (s) [aumentado significativamente]

# Intersecção (mais tolerante - banda mais baixa para detectar interseções mais cedo)
INT_BAND_Y0_FRAC        = 0.25
INT_BAND_Y1_FRAC        = 0.98
INT_STABLE_FR           = 2
INT_MATCH_RADIUS        = 24
INTERSECTION_COOLDOWN   = 1.0

# Início cego (linha horizontal)
ROW_BAND_TOP_FRAC       = 0.45
ROW_BAND_BOTTOM_FRAC    = 0.85
ROW_PEAK_FRAC_THR       = 0.020
LOSE_FRAMES_START       = 3
START_TIMEOUT_S         = 6.0

# ============================ VISÃO (robot2) ============================
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
    h, w = imagem.shape[:2]
    cx_img = w // 2
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
    v_esq = int(np.clip(v_esq, 80, VELOCIDADE_MAX))  # Mínimo aumentado para 80
    v_dir = int(np.clip(v_dir, 80, VELOCIDADE_MAX))  # Mínimo aumentado para 80
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
    drive_cap(arduino, START_SPEED, START_SPEED); time.sleep(0.1)
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

            drive_cap(arduino, START_SPEED, START_SPEED)
            if not saw:
                if present: saw=True; lost=0
            else:
                if present: lost=0
                else:
                    lost+=1
                    if lost>=LOSE_FRAMES_START:
                        drive_cap(arduino,0,0); return True
            if (time.time()-t0)>START_TIMEOUT_S:
                drive_cap(arduino,0,0); return False
            raw.truncate(0); raw.seek(0)
    finally:
        raw.truncate(0)

def spin_in_place_until_seen(arduino, camera, side_hint='L'):
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    turn_sign = -1 if side_hint=='L' else +1
    seen_cnt=0; t0=time.time()
    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            img=f.array
            _, err, conf = processar_imagem(img)
            v_esq, v_dir = turn_sign*PIVOT_MIN, -turn_sign*PIVOT_MIN
            drive_cap(arduino, v_esq, v_dir, cap=PIVOT_CAP)

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
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    stable=0; t0=time.time(); lost_frames=0; last_err=0.0; state='FOLLOW'
    frame_count = 0
    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            frame_count += 1
            img=f.array
            _, erro, conf = processar_imagem(img)

            if conf==1:
                state='FOLLOW'; lost_frames=0; last_err=erro
                v_esq, v_dir = calcular_velocidades_auto(erro, ALIGN_BASE)
                print(f"      Frame {frame_count}: Seguindo | erro={erro:.1f} | vel=({v_esq},{v_dir})")
            else:
                lost_frames+=1
                if lost_frames>=LOST_MAX_FRAMES:
                    state='LOST'
                if state=='LOST':
                    turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                    v_esq, v_dir = int(turn*0.7), int(-turn*0.7)  # giro suave
                    print(f"      Frame {frame_count}: Perdido! Procurando | vel=({v_esq},{v_dir})")
                else:
                    v_esq, v_dir = ALIGN_BASE, ALIGN_BASE
                    print(f"      Frame {frame_count}: Sem linha | vel=reto")

            drive_cap(arduino, v_esq, v_dir, cap=ALIGN_CAP)

            if conf==1 and abs(erro)<=ALIGN_TOL_PIX:
                stable+=1
                print(f"      → Estável: {stable}/{ALIGN_STABLE} frames")
            else:
                stable=0

            if stable>=ALIGN_STABLE:
                print(f"      ✅ Alinhamento concluído após {frame_count} frames!")
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
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    last_err=0.0; lost_frames=0; state='FOLLOW'
    last_int_t=0.0; prev=None; stable=0; t0=time.time()
    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            img=f.array
            mask=build_binary_mask(img)
            _,erro,conf = processar_imagem(img)

            if conf == 1:
                state='FOLLOW'; lost_frames=0; last_err=erro
                speed_scale = max(0.35, 1.0 - abs(erro) / float(E_MAX_PIX))
                base_speed  = int(np.clip(VELOCIDADE_BASE * speed_scale, V_MIN, VELOCIDADE_MAX))
                v_esq, v_dir = calcular_velocidades_auto(erro, base_speed)
            else:
                lost_frames += 1
                if lost_frames >= LOST_MAX_FRAMES:
                    state='LOST'
                if state=='LOST':
                    turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                    v_esq, v_dir = int(turn), int(-turn)
                else:
                    base_speed = int(np.clip(VELOCIDADE_BASE * 0.35, V_MIN, VELOCIDADE_MAX))
                    v_esq, v_dir = calcular_velocidades_auto(0, base_speed)

            drive_cap(arduino, v_esq, v_dir, cap=ALIGN_CAP)  # manter razoável

            pts, detected_lines = detect_intersections(mask)
            h=mask.shape[0]
            y0=int(h*INT_BAND_Y0_FRAC); y1=int(h*INT_BAND_Y1_FRAC)
            cand = best_intersection_in_band(pts, h, y0, y1)

            # Debug: mostra interseções detectadas
            if pts:
                print(f"   🔍 Detectadas {len(pts)} interseções: {[f'({x},{y})' for x,y in pts]}")
                if cand:
                    print(f"   🎯 Candidata na banda: ({cand[0]},{cand[1]})")
                else:
                    print(f"   ⚠️  Nenhuma interseção na banda y∈[{y0},{y1}]")

            # Visualização das interseções (igual ao robot_new.py)
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
            for idx, (x, y) in enumerate(pts, 1):
                cv2.circle(display_frame, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(display_frame, (x, y), 12, (255, 255, 255), 2)
                cv2.putText(display_frame, f"{idx}", (x + 15, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Destaque a interseção candidata na banda
            if cand is not None:
                cv2.circle(display_frame, cand, 15, (255, 0, 255), 3)  # magenta para destacar

            # HUD igual ao robot_new.py
            cv2.putText(display_frame, f"State: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display_frame, f"Lines: {len(detected_lines)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(display_frame, f"Intersections: {len(pts)}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(display_frame, f"Stable: {stable}/{INT_STABLE_FR}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)
            cv2.putText(display_frame, f"Band: {INT_BAND_Y0_FRAC:.2f}-{INT_BAND_Y1_FRAC:.2f}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            # Enviar frame processado para streaming
            send_frame_to_stream(display_frame)

            if cand is not None:
                if prev is not None and (np.hypot(cand[0]-prev[0], cand[1]-prev[1]) <= INT_MATCH_RADIUS):
                    stable += 1
                else:
                    prev = cand; stable = 1
            else:
                prev=None; stable=0

            now=time.time()
            if stable>=INT_STABLE_FR and (now-last_int_t)>=INTERSECTION_COOLDOWN:
                drive_cap(arduino, 80, 80, cap=ALIGN_CAP); time.sleep(0.10); drive_cap(arduino,0,0)
                last_int_t=now; return True

            raw.truncate(0); raw.seek(0)
            if (now-t0)>10.0:
                drive_cap(arduino,0,0); return False
    finally:
        raw.truncate(0)

# ====================== Planejamento e Execução ======================
def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def front_left_right_corners(sx,sy,orient):
    if orient==0:  return ( (sx,sy),     (sx+1,sy) )
    if orient==1:  return ( (sx+1,sy),   (sx+1,sy+1) )
    if orient==2:  return ( (sx+1,sy+1), (sx,sy+1) )
    if orient==3:  return ( (sx,sy+1),   (sx,sy) )
    raise ValueError

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
    if b[1]<a[1]: return 0
    if b[1]>a[1]: return 2
    if b[0]>a[0]: return 1
    return 3
def relative_turn(cur_dir,want_dir): return {0:'F',1:'R',2:'U',3:'L'}[(want_dir-cur_dir)%4]

def dir_name(d):
    return {0:'Norte', 1:'Leste', 2:'Sul', 3:'Oeste'}[d]

def leave_square_to_best_corner(arduino, camera, sx, sy, cur_dir, target):
    """
    Sai do quadrado usando a orientação declarada (assumida correta).
    """
    print(f"🚶 Saindo do quadrado ({sx},{sy})")
    print(f"   Orientação: {dir_name(cur_dir)}")
    print(f"   Destino: {target}")

    left_corner, right_corner = front_left_right_corners(sx, sy, cur_dir)
    dl = manhattan(left_corner, target)
    dr = manhattan(right_corner, target)
    side_hint = 'L' if dl <= dr else 'R'
    chosen = left_corner if side_hint=='L' else right_corner

    print(f"   Escolhendo canto {chosen} (virada: {'esquerda' if side_hint=='L' else 'direita'})")
    print(f"   Distância Manhattan: {dl} vs {dr}")

    # Reta cega
    if not straight_until_seen_then_lost(arduino, camera):
        print("✗ Falha na reta inicial.")
        return None, None, False

    # Pivô: girar até ver + alinhar andando
    if not spin_in_place_until_seen(arduino, camera, side_hint=side_hint):
        print("✗ Falha no pivô (não viu linha).")
        return None, None, False

    if not forward_align_on_line(arduino, camera):
        print("✗ Falha no alinhamento após pivô.")
        return None, None, False

    # Segue para 1ª intersecção
    if not go_to_next_intersection(arduino, camera):
        print("✗ Falha ao alcançar a intersecção.")
        return None, None, False

    # ⚠️  IMPORTANTE: Executa o giro final para a direção correta
    new_dir = (cur_dir - 1) % 4 if side_hint == 'L' else (cur_dir + 1) % 4
    print(f"🔄 Executando giro final: {dir_name(cur_dir)} → {dir_name(new_dir)}")

    # Para antes de girar
    drive_cap(arduino, 0, 0); time.sleep(0.2)

    # Calcula e executa o giro
    rel_turn = relative_turn(cur_dir, new_dir)
    exec_turn(arduino, rel_turn)

    print(f"✅ Giro final executado - Agora virado para {dir_name(new_dir)}")
    return chosen, new_dir, True

def exec_turn(arduino, rel):
    if rel=='F': return
    if rel=='L':
        drive_cap(arduino, -TURN_SPEED, TURN_SPEED, cap=ALIGN_CAP); time.sleep(0.75); drive_cap(arduino,0,0); time.sleep(0.10)
    elif rel=='R':
        drive_cap(arduino, TURN_SPEED, -TURN_SPEED, cap=ALIGN_CAP); time.sleep(0.75); drive_cap(arduino,0,0); time.sleep(0.10)
    else:
        drive_cap(arduino, TURN_SPEED, -TURN_SPEED, cap=ALIGN_CAP); time.sleep(1.50); drive_cap(arduino,0,0); time.sleep(0.10)

def follow_path(arduino, start_node, start_dir, path, camera):
    """
    O robô JÁ ESTÁ na primeira interseção (start_node) após leave_square_to_best_corner.
    Esta função executa o resto do caminho A*.
    """
    cur_node=start_node; cur_dir=start_dir
    drive_cap(arduino,0,0); time.sleep(0.1)

    # Mostra o caminho completo
    print("🗺️  CAMINHO CALCULADO PELO A*:")
    path_str = " → ".join([f"({x},{y})" for x,y in path])
    print(f"   {path_str}")
    print(f"   Total: {len(path)-1} movimentos")
    print()

    # Verifica se já estamos no destino
    if len(path) == 1 and path[0] == start_node:
        print(f"🎯 Já estamos no destino ({start_node[0]},{start_node[1]})!")
        return cur_node,cur_dir,True

    # SEMPRE vai para a primeira interseção do caminho para garantir posicionamento
    first_target = path[0]
    print(f"🎯 Confirmando posição na interseção ({first_target[0]},{first_target[1]})")

    # Tenta ir para a primeira interseção (deve ser rápida se já estiver lá)
    if not go_to_next_intersection(arduino, camera):
        print(f"   ❌ Falha ao confirmar posição em ({first_target[0]},{first_target[1]})")
        return cur_node,cur_dir,False
    print(f"   ✅ Posição confirmada em ({first_target[0]},{first_target[1]})")
    cur_node = first_target
    print()

    # Executa cada passo do caminho (começando do segundo nó)
    for i in range(1,len(path)):
        nxt=path[i]
        want=orientation_of_step(cur_node, nxt)
        rel=relative_turn(cur_dir,want)

        # Mostra cada virada específica
        turn_names = {'F':'Frente', 'L':'Esquerda (90°)', 'R':'Direita (90°)', 'U':'Meia-volta (180°)'}
        print(f"🔄 Intersecção ({cur_node[0]},{cur_node[1]}): {dir_name(cur_dir)} → {turn_names[rel]} → {dir_name(want)}")
        print(f"   Indo para ({nxt[0]},{nxt[1]})")

        # ⚠️  IMPORTANTE: Para completamente antes de virar
        drive_cap(arduino, 0, 0); time.sleep(0.2)
        print(f"   🛑 Parado para executar giro")

        exec_turn(arduino, rel); cur_dir=want
        print(f"   ✅ Giro executado")
        if not go_to_next_intersection(arduino, camera):
            print(f"   ❌ Falha ao alcançar ({nxt[0]},{nxt[1]})")
            return cur_node,cur_dir,False
        print(f"   ✅ Chegou em ({nxt[0]},{nxt[1]})")
        print()
        cur_node=nxt

    print(f"🎯 Chegou ao destino final!")
    return cur_node,cur_dir,True

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

    try:
        print(f"🏠 INÍCIO: Quadrado ({sx},{sy}), Orientação {dir_name(cur_dir)}")
        print(f"📦 DESTINO: Nó ({tx},{ty})")
        print()

        # Frame de início
        send_basic_frame(camera, f"Quadrado ({sx},{sy}) -> No ({tx},{ty})")

        start_node, cur_dir, ok = leave_square_to_best_corner(arduino, camera, sx, sy, cur_dir, target)
        if not ok: print("❌ Falha na saída."); return

        print(f"📍 Após saída: Posição {start_node}, Direção {dir_name(cur_dir)}")
        print()

        print("🤖 EXECUTANDO A* PARA CALCULAR CAMINHO...")
        send_basic_frame(camera, "Calculando caminho A*...")

        path=a_star(start_node, target, GRID_NODES)
        if path is None:
            print("❌ Nenhum caminho encontrado pelo A*.")
            send_basic_frame(camera, "ERRO: Caminho nao encontrado!")
            return

        print()
        send_basic_frame(camera, f"Caminho: {' -> '.join([f'({x},{y})' for x,y in path])}")
        _,cur_dir,ok=follow_path(arduino, start_node, cur_dir, path, camera)
        if not ok: print("❌ Falha na ida."); return
        print("✅ Entrega realizada com sucesso!")
        print()

        if not args.no_return:
            print("🔄 CALCULANDO CAMINHO DE RETORNO...")
            back=a_star(target, start_node, GRID_NODES)
            if back is None:
                print("❌ Nenhum caminho de retorno encontrado.")
                return
            print()
            _,_,ok=follow_path(arduino, target, cur_dir, back, camera)
            print("✅ Retornou ao ponto inicial!" if ok else "❌ Falhou no retorno.")
    finally:
        try:
            enviar_comando_motor_serial(arduino, 0, 0)
            arduino.write(b'a\n'); arduino.close()
        except Exception: pass
        camera.close()

if __name__=='__main__':
    main()
