# delivery_delivery_v12.py — Delivery 4x4 (robot2 vision/control inline)
# Novidades v12:
#  • Pós-pivô em 2 fases: (a) girar até VER a linha; (b) avançar devagar usando P até CENTRALIZAR.
#    Isso resolve o “não anda o suficiente depois do pivô” e melhora o lock.
#  • Intersecções menos “estritas”: banda mais baixa e estabilidade em 2 frames.
#  • Mantém todas as rotinas de visão/controle idênticas ao robot2.py.
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2, time, numpy as np, serial, argparse

# ============================= PARÂMETROS (iguais ao robot2) =============================
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 180
HOUGHP_THRESHOLD    = 35
HOUGHP_MINLEN_FRAC  = 0.35
HOUGHP_MAXGAP       = 20
ROI_CROP_FRAC       = 0.20
RHO_MERGE           = 40
THETA_MERGE_DEG     = 6
ORTH_TOL_DEG        = 15
PAR_TOL_DEG         = 8

VELOCIDADE_BASE = 150
VELOCIDADE_CURVA = 100
Kp = 0.75
VELOCIDADE_MAX = 255
E_MAX_PIX       = IMG_WIDTH // 2
V_MIN           = 0
SEARCH_SPEED    = 120
LOST_MAX_FRAMES = 5
DEAD_BAND       = 6
ROI_BOTTOM_FRAC = 0.55
MIN_AREA_FRAC   = 0.004
MAX_AREA_FRAC   = 0.25
ASPECT_MIN      = 2.0
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
ALIGN_BASE      = 90      # velocidade base na fase de alinhamento (P)  [<=100]
ALIGN_CAP       = 140     # cap de segurança na fase de alinhamento
ALIGN_TOL_PIX   = 8       # centralização final
ALIGN_STABLE    = 4       # frames estáveis
ALIGN_TIMEOUT   = 2.0     # tempo máx. alinhando (s)

# Intersecção (mais tolerante)
INT_BAND_Y0_FRAC        = 0.58
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
    v_esq = int(np.clip(v_esq, 15, VELOCIDADE_MAX))
    v_dir = int(np.clip(v_dir, 15, VELOCIDADE_MAX))
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
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    stable=0; t0=time.time(); lost_frames=0; last_err=0.0; state='FOLLOW'
    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            img=f.array
            _, erro, conf = processar_imagem(img)
            if conf==1:
                state='FOLLOW'; lost_frames=0; last_err=erro
                v_esq, v_dir = calcular_velocidades_auto(erro, ALIGN_BASE)
            else:
                lost_frames+=1
                if lost_frames>=LOST_MAX_FRAMES:
                    state='LOST'
                if state=='LOST':
                    turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                    v_esq, v_dir = int(turn*0.7), int(-turn*0.7)  # giro suave
                else:
                    v_esq, v_dir = ALIGN_BASE, ALIGN_BASE

            drive_cap(arduino, v_esq, v_dir, cap=ALIGN_CAP)

            if conf==1 and abs(erro)<=ALIGN_TOL_PIX:
                stable+=1
            else:
                stable=0
            if stable>=ALIGN_STABLE:
                drive_cap(arduino, 70, 70, cap=ALIGN_CAP); time.sleep(0.10)
                drive_cap(arduino, 0, 0); return True

            if (time.time()-t0) > ALIGN_TIMEOUT:
                drive_cap(arduino,0,0); return False

            raw.truncate(0); raw.seek(0)
    finally:
        raw.truncate(0)

def best_intersection_in_band(pts, h, band_y0, band_y1):
    cand=None; best_y=-1
    for (x,y) in pts:
        if band_y0<=y<=band_y1 and y>best_y:
            best_y=y; cand=(x,y)
    return cand

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

            pts,_ = detect_intersections(mask)
            h=mask.shape[0]
            y0=int(h*INT_BAND_Y0_FRAC); y1=int(h*INT_BAND_Y1_FRAC)
            cand = best_intersection_in_band(pts, h, y0, y1)

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

def leave_square_to_best_corner(arduino, camera, sx, sy, cur_dir, target):
    left_corner, right_corner = front_left_right_corners(sx, sy, cur_dir)
    dl = manhattan(left_corner, target); dr = manhattan(right_corner, target)
    side_hint = 'L' if dl<=dr else 'R'
    chosen = left_corner if side_hint=='L' else right_corner
    print(f"→ Saindo para {chosen} (pivô 2-fases; hint='{side_hint}')")

    # Reta cega
    if not straight_until_seen_then_lost(arduino, camera):
        print("✗ Falha na reta inicial."); return None,None,False
    # Pivô: girar até ver + alinhar andando
    if not spin_in_place_until_seen(arduino, camera, side_hint=side_hint):
        print("✗ Falha no pivô (não viu linha)."); return None,None,False
    if not forward_align_on_line(arduino, camera):
        print("✗ Falha no alinhamento após pivô."); return None,None,False
    # Segue para 1ª intersecção
    if not go_to_next_intersection(arduino, camera):
        print("✗ Falha ao alcançar a intersecção."); return None,None,False

    new_dir = (cur_dir - 1)%4 if side_hint=='L' else (cur_dir + 1)%4
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
    cur_node=start_node; cur_dir=start_dir
    drive_cap(arduino,0,0); time.sleep(0.1)
    for i in range(1,len(path)):
        nxt=path[i]
        want=orientation_of_step(cur_node, nxt)
        rel=relative_turn(cur_dir,want)
        exec_turn(arduino, rel); cur_dir=want
        if not go_to_next_intersection(arduino, camera):
            return cur_node,cur_dir,False
        cur_node=nxt
    return cur_node,cur_dir,True

# =================================== MAIN ===================================
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--square', type=int, nargs=2, required=True, metavar=('SX','SY'))
    ap.add_argument('--orient', type=str, required=True, help='N/E/S/W')
    ap.add_argument('--target', type=int, nargs=2, required=True, metavar=('TX','TY'))
    ap.add_argument('--no-return', action='store_true')
    return ap.parse_args()

def main():
    args=parse_args()
    sx,sy=args.square; tx,ty=args.target
    orient=args.orient.strip().upper()
    if orient not in ('N','E','S','W','O'): raise SystemExit("orient deve ser N/E/S/W")
    cur_dir={'N':0,'E':1,'S':2,'W':3,'O':3}[orient]
    if not (0<=sx<=3 and 0<=sy<=3): raise SystemExit("square 0..3 0..3")
    if not (0<=tx<=4 and 0<=ty<=4): raise SystemExit("target 0..4 0..4")
    target=(tx,ty)

    camera = PiCamera(); camera.resolution=(IMG_WIDTH, IMG_HEIGHT); camera.framerate=24
    time.sleep(0.6)  # warm-up

    arduino = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=1); time.sleep(2)
    try:
        arduino.write(b'A10\n')
        try: print("Arduino:", arduino.readline().decode('utf-8').strip())
        except Exception: pass
    except Exception: pass

    try:
        start_node, cur_dir, ok = leave_square_to_best_corner(arduino, camera, sx, sy, cur_dir, target)
        if not ok: print("Falha na saída."); return
        path=a_star(start_node, target, GRID_NODES)
        if path is None: print("Sem caminho."); return
        _,cur_dir,ok=follow_path(arduino, start_node, cur_dir, path, camera)
        if not ok: print("Falha na ida."); return
        print("✅ Entrega ok.")
        if not args.no_return:
            back=a_star(target, start_node, GRID_NODES)
            _,_,ok=follow_path(arduino, target, cur_dir, back, camera)
            print("✅ Retornou." if ok else "✗ Falhou no retorno.")
    finally:
        try:
            enviar_comando_motor_serial(arduino, 0, 0)
            arduino.write(b'a\n'); arduino.close()
        except Exception: pass
        camera.close()

if __name__=='__main__':
    main()
