# delivery_delivery_v12.py - Delivery 4x4 (robot2 vision/control inline)
# What's new in v12:
#  - Two-phase post-pivot: (a) turn until the line is SEEN; (b) advance slowly using P until CENTERED.
#    This solves the "does not move enough after pivot" issue and improves locking.
#  - Less "strict" intersections: lower band and stability in 2 frames.
#  - Keeps all vision/control routines identical to robot2.py.
from picamera.array import PiRGBArray
from picamera import PiCamera
from picamera.exc import PiCameraValueError
import cv2, time, numpy as np, serial, argparse, zmq, base64

# --- REMOTE CONTROL CONFIGURATION ---
SERVER_IP = "192.168.137.22"  # IP of the computer running server.py
# --- END CONFIGURATION ---

# ============================= PARAMETERS (same as robot2) =============================
IMG_WIDTH, IMG_HEIGHT = 320, 240
THRESHOLD_VALUE = 160  # Fine tuning for better detection
HOUGHP_THRESHOLD    = 35
HOUGHP_MINLEN_FRAC  = 0.35
HOUGHP_MAXGAP       = 20
ROI_CROP_FRAC       = 0.20
RHO_MERGE           = 40
THETA_MERGE_DEG     = 6
ORTH_TOL_DEG        = 15
PAR_TOL_DEG         = 8

DEFAULT_SPEED_LEVEL = 0.85  # Global adjustment (e.g., 0.5 = eco, 1.0 = standard, 2.0 = boost)
speed = DEFAULT_SPEED_LEVEL

BASE_VELOCIDADE_BASE = 120
BASE_VELOCIDADE_CURVA = 120
BASE_SEARCH_SPEED = 110
BASE_START_SPEED = 90
BASE_ALIGN_BASE = 100
BASE_ALIGN_CAP = 135
BASE_PIVOT_MIN = 150
BASE_PIVOT_CAP = 150
BASE_TURN_SPEED = 150
BASE_STRAIGHT_SPEED = 110
BASE_CRAWL_SPEED = 95
BASE_CRAWL_DURATION = 0.07
BASE_BORDER_CRAWL_DURATION = 0.06
BASE_TURN_DURATION = 0.85
BASE_UTURN_RATIO = 1.75  # Factor applied to the 90-degree turn duration (adjustable)
BASE_APPROACH_FLOOR = 100
BASE_APPROACH_LOST = 110
BASE_CELEBRATION_WIGGLE = 130

def _compute_base_speed(level):
    lvl = max(level, 0.5)
    if lvl <= 1.0:
        # Interpolate from 0.5->90 to 1.0->BASE_VELOCIDADE_BASE
        return int(round(90 + (lvl - 0.5) * (BASE_VELOCIDADE_BASE - 90) / 0.5))
    # Above 1.0 it grows with a gentle slope (80 per unit)
    return int(round(BASE_VELOCIDADE_BASE + (lvl - 1.0) * 80))

VELOCIDADE_BASE = _compute_base_speed(speed)
speed_multiplier = VELOCIDADE_BASE / float(BASE_VELOCIDADE_BASE)

def _scale_speed(value, exponent=1.0, min_value=None, max_value=None):
    scaled = value * (speed_multiplier ** exponent)
    if min_value is not None:
        scaled = max(min_value, scaled)
    if max_value is not None:
        scaled = min(max_value, scaled)
    return int(round(scaled))

VELOCIDADE_CURVA = _scale_speed(BASE_VELOCIDADE_CURVA)
Kp = 1.0              # Ganho do controlador P - aumentado para melhor controle
VELOCIDADE_MAX = 255
E_MAX_PIX       = IMG_WIDTH // 2
V_MIN           = 0
SEARCH_SPEED    = _scale_speed(BASE_SEARCH_SPEED, max_value=VELOCIDADE_MAX)
LOST_MAX_FRAMES = 5
DEAD_BAND       = 3
ROI_BOTTOM_FRAC = 0.55
MIN_AREA_FRAC   = 0.003  # Ainda mais reduzido para detectar linhas
MAX_AREA_FRAC   = 0.3    # Um pouco mais tolerante
ASPECT_MIN      = 2.0    # Ainda menos rigoroso
LINE_POLARITY   = 'white'               # Forced back to white
USE_ADAPTIVE    = False

PORTA_SERIAL = '/dev/ttyACM0'
BAUDRATE = 115200

# ======== DELIVERY (extra) ========
GRID_NODES = (5, 5)       # 4x4 squares -> 5x5 nodes
START_SPEED  = _scale_speed(BASE_START_SPEED, max_value=VELOCIDADE_MAX)  # reta cega
TURN_SPEED   = _scale_speed(BASE_TURN_SPEED, max_value=VELOCIDADE_MAX)   # giros 90/180 ajustados

# Obstacle management
blocked_edges = set()  # Set of tuples (node1, node2) representing blocked edges


def is_border_node(node):
    """Return True if the intersection belongs to the outer edge of the 5x5 grid."""
    if node is None:
        return False
    x, y = node
    max_x, max_y = GRID_NODES[0] - 1, GRID_NODES[1] - 1
    return x == 0 or y == 0 or x == max_x or y == max_y

# Pivot and post-pivot acquisition
PIVOT_CAP       = _scale_speed(BASE_PIVOT_CAP, max_value=VELOCIDADE_MAX)
PIVOT_MIN       = _scale_speed(BASE_PIVOT_MIN, min_value=80)
PIVOT_TIMEOUT   = 2.0   # Timeout for pivot (increased for reliability)
SEEN_FRAMES     = 1       # Consecutive frames "seeing" the line before exiting the turn - reduced
ALIGN_BASE      = _scale_speed(BASE_ALIGN_BASE, min_value=70)
ALIGN_CAP       = _scale_speed(BASE_ALIGN_CAP, max_value=VELOCIDADE_MAX)
# Alignment parameters
ALIGN_TOL_PIX   = 8       # Final centering
ALIGN_STABLE    = 2       # Stable frames [reduced to enter FOLLOW faster]
ALIGN_TIMEOUT   = 6.0     # Maximum alignment time (s) [significantly increased]

# Intersection (parameters from robot_pedro.py - more robust)
Y_START_SLOWING_FRAC = 0.60  # Start slowing when the intersection passes 70% of the height
Y_TARGET_STOP_FRAC = 0.92    # Stop slightly before the bottom limit
CRAWL_SPEED = _scale_speed(BASE_CRAWL_SPEED, min_value=70, max_value=VELOCIDADE_MAX)
speed_multiplier_for_time = max(speed_multiplier, 0.7)
CRAWL_DURATION_S = BASE_CRAWL_DURATION / speed_multiplier_for_time
BORDER_CRAWL_DURATION_S = BASE_BORDER_CRAWL_DURATION / speed_multiplier_for_time
turn_speed_gain = max(TURN_SPEED / float(BASE_TURN_SPEED), 0.1)
TURN_DURATION_S = float(np.clip(BASE_TURN_DURATION / turn_speed_gain, 0.35, 1.2))
UTURN_DURATION_S = float(np.clip((BASE_UTURN_RATIO * TURN_DURATION_S), 0.8, 2.3))
STRAIGHT_SPEED = _scale_speed(BASE_STRAIGHT_SPEED, max_value=VELOCIDADE_MAX)
STRAIGHT_DURATION_S = 0.5    # Duration (seconds) to cross
BORDER_MARGIN_FRAC = 0.12    # Lateral fraction considered the grid border
BORDER_Y_START_SLOWING_FRAC = 0.45  # Border ROI starts earlier (intersections vanish sooner)
BORDER_Y_TARGET_STOP_FRAC = 0.74    # Higher target: borders disappear well before the bottom limit
INTERSECTION_MEMORY_S = 0.70        # Time in seconds to keep an intersection alive after it disappears
INTERSECTION_MEMORY_GROW_FRAC_PER_S = 0.70  # Height fraction projected per second when relying on memory
INTERSECTION_MEMORY_EXTRA_FRAC = 0.012        # Additional limit (as height fraction) above the last real Y
BORDER_INTERSECTION_MEMORY_EXTRA_FRAC = 0.028  # Larger extra limit on borders
INTERSECTION_BORDER_STOP_PAD_FRAC = 0.020     # Additional margin above the last Y when using memory
INTERSECTION_DESCENT_MIN_FRAMES = 5          # Minimum frames seeing the intersection descend
INTERSECTION_DESCENT_TOL_PX = 6             # Tolerance for small Y oscillations
INTERSECTION_DESCENT_MIN_DELTA_FRAC = 0.08   # Minimum descent (height fraction) to trust long memory
INTERSECTION_DESCENT_MEMORY_S = 1.4          # Extra memory time when the descent is confirmed
INTERSECTION_REJECT_JUMP_FRAC = 0.05         # If a new intersection jumps up too much, ignore it and keep the old one
APPROACH_TIMEOUT_S = 2.5            # Maximum time stuck in APPROACHING before forcing a stop
APPROACH_FLOOR_SPEED = _scale_speed(BASE_APPROACH_FLOOR, min_value=80, max_value=VELOCIDADE_MAX)
APPROACH_LOST_SPEED = max(APPROACH_FLOOR_SPEED + 5,
                           _scale_speed(BASE_APPROACH_LOST, min_value=APPROACH_FLOOR_SPEED + 5,
                                        max_value=VELOCIDADE_MAX))
CELEBRATION_WIGGLE_SPEED = _scale_speed(BASE_CELEBRATION_WIGGLE, max_value=VELOCIDADE_MAX)

# Blind start (horizontal line)
ROW_BAND_TOP_FRAC       = 0.45
ROW_BAND_BOTTOM_FRAC    = 0.85
ROW_PEAK_FRAC_THR       = 0.030
LOSE_FRAMES_START       = 5
START_TIMEOUT_S         = 6.0

# ============================ VISION (robot2) ============================
def _angle_diff(a, b):
    d = abs((a - b) % np.pi)
    return min(d, np.pi - d)

def _deg(x): return np.deg2rad(x)

def distance_to_line(point, line):
    """Compute the distance from a point (x, y) to a line (rho, theta)."""
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

    # Simple morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove upper part of the image (sky/noise)
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

    # Ignore initial detections to avoid picking up the floor/noise
    initial_frames_ignored += 1
    if initial_frames_ignored <= IGNORE_INITIAL_FRAMES:
        return imagem, 0, 0  # Return without detection

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

    # Polarity selection (same as robot_pedro.py)
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
    # Allow lower speeds for corrections, but keep a reasonable minimum
    v_esq = int(np.clip(v_esq, 60, VELOCIDADE_MAX))
    v_dir = int(np.clip(v_dir, 60, VELOCIDADE_MAX))
    return v_esq, v_dir

def enviar_comando_motor_serial(arduino, v_esq, v_dir):
    comando = f"C {v_dir} {v_esq}"
    arduino.write(comando.encode('utf-8'))
    # Check for obstacle response
    time.sleep(0.01)  # Small delay to receive response
    if arduino.in_waiting > 0:
        response = arduino.readline().decode('utf-8').strip()
        if response == "OB":
            return True  # Obstacle detected
    return False  # No obstacle

# ====================== Utilidades ======================
def drive_cap(arduino, v_esq, v_dir, cap=255):
    v_esq=int(np.clip(v_esq, -cap, cap))
    v_dir=int(np.clip(v_dir, -cap, cap))
    obstacle_detected = enviar_comando_motor_serial(arduino, v_esq, v_dir)
    return obstacle_detected

def celebrate_delivery(arduino, duration=3.0, wiggle_speed=None, pause=0.3):
    """Perform a brief wiggle without moving the robot to signal a delivery."""
    if wiggle_speed is None:
        wiggle_speed = CELEBRATION_WIGGLE_SPEED
    print("Starting delivery celebration...")
    t_end = time.time() + duration
    toggle = True
    while time.time() < t_end:
        if toggle:
            _ = drive_cap(arduino, wiggle_speed, -wiggle_speed)
        else:
            _ = drive_cap(arduino, -wiggle_speed, wiggle_speed)
        toggle = not toggle
        time.sleep(pause)
    _ = drive_cap(arduino, 0, 0); time.sleep(0.2)
    print("Celebration finished.")

def handle_obstacle_uturn(arduino, camera):
    """Execute U-turn when obstacle is detected."""
    print("!!! OBSTACLE DETECTED - Executing U-turn !!!")
    
    # Disable IR protection
    arduino.write(b'I0')
    time.sleep(0.1)
    
    # Execute U-turn
    drive_cap(arduino, TURN_SPEED, -TURN_SPEED)
    time.sleep(UTURN_DURATION_S)
    drive_cap(arduino, 0, 0)
    time.sleep(0.5)
    
    # Re-enable IR protection
    arduino.write(b'I1')
    time.sleep(0.1)
    
    print("U-turn complete, IR protection re-enabled")

def add_blocked_edge(node1, node2):
    """Add a blocked edge to the global set (bidirectional)."""
    global blocked_edges
    # Normalize edge representation (smaller node first)
    edge = tuple(sorted([node1, node2]))
    blocked_edges.add(edge)
    print(f"Added blocked edge: {node1} <-> {node2}")

def is_edge_blocked(node1, node2):
    """Check if an edge is blocked."""
    edge = tuple(sorted([node1, node2]))
    return edge in blocked_edges

# ====================== Blind start / Pivot (2 phases) / Intersection ======================
def straight_until_seen_then_lost(arduino, camera):
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    saw=False; lost=0; t0=time.time()

    # Start with a higher speed to cover more distance
    initial_speed = int(START_SPEED * 1.15)  # 15% faster (less than before)
    _ = drive_cap(arduino, initial_speed, initial_speed); time.sleep(0.1)

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

            # Send frame to the stream during the initial straight segment
            display_frame = img.copy()
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)
            cv2.putText(display_frame, f"Initial Straight - Present: {present}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            send_frame_to_stream(display_frame)

            # Keep the initial speed until the line is seen
            current_speed = initial_speed if not saw else START_SPEED
            _ = drive_cap(arduino, current_speed, current_speed)  # Ignore obstacle detection during initial straight

            if not saw:
                if present: saw=True; lost=0
            else:
                if present: lost=0
                else:
                    lost+=1
                    if lost>=LOSE_FRAMES_START:
                        # After losing the line, move a little further forward
                        _ = drive_cap(arduino, START_SPEED, START_SPEED); time.sleep(0.15)  # Reduced time
                        _ = drive_cap(arduino,0,0); return True
            if (time.time()-t0)>START_TIMEOUT_S:
                _ = drive_cap(arduino,0,0); return False
            raw.truncate(0); raw.seek(0)
    finally:
        raw.truncate(0)

def spin_in_place_until_seen(arduino, camera, side_hint='L', orient=0):
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    turn_sign = -1 if side_hint=='L' else +1
    # Direction adjustments per orientation can be added here if needed
    # For now, every orientation uses the standard logic
    pass
    seen_cnt=0; t0=time.time()
    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            img=f.array
            img_display, err, conf = processar_imagem(img)
            v_esq, v_dir = turn_sign*PIVOT_MIN, -turn_sign*PIVOT_MIN
            _ = drive_cap(arduino, v_esq, v_dir, cap=PIVOT_CAP)  # Ignore obstacle detection during pivot

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
                _ = drive_cap(arduino, 0, 0)
                return True

            if (time.time()-t0) > PIVOT_TIMEOUT:
                _ = drive_cap(arduino,0,0); return False
            raw.truncate(0); raw.seek(0)
    finally:
        raw.truncate(0)

def forward_align_on_line(arduino, camera):
    """Move forward slowly using P until the error stays small for a few frames."""
    print("   Starting line alignment...")
    # Small pause to stabilize after the pivot
    time.sleep(0.3)
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    stable=0; t0=time.time(); lost_frames=0; last_err=0.0; state='FOLLOW'
    frame_count = 0; last_frame_sent = 0

    # Larger tolerance at the start (post-pivot may be unstable)
    initial_tolerance_frames = 10
    try:
        for f in camera.capture_continuous(raw, format="bgr", use_video_port=True):
            frame_count += 1
            img=f.array
            _, erro, conf = processar_imagem(img)

            # Higher tolerance during the first frames after the pivot
            effective_lost_max = LOST_MAX_FRAMES * 2 if frame_count <= initial_tolerance_frames else LOST_MAX_FRAMES

            if conf==1:
                state='FOLLOW'; lost_frames=0; last_err=erro
                v_esq, v_dir = calcular_velocidades_auto(erro, ALIGN_BASE)
                print(f"      Frame {frame_count}: Following | error={erro:.1f} | vel=({v_esq},{v_dir})")
            else:
                lost_frames+=1
                if lost_frames>=effective_lost_max:
                    state='LOST'
                if state=='LOST':
                    turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                    v_esq, v_dir = int(turn*0.7), int(-turn*0.7)  # gentle turn
                    print(f"      Frame {frame_count}: Lost! Searching | vel=({v_esq},{v_dir})")
                else:
                    v_esq, v_dir = ALIGN_BASE, ALIGN_BASE
                    print(f"      Frame {frame_count}: No line | vel=straight (tolerance: {effective_lost_max})")

            _ = drive_cap(arduino, v_esq, v_dir, cap=ALIGN_CAP)  # Ignore obstacle detection during alignment

            # Create frame for visualization
            display_frame = img.copy()
            mask = build_binary_mask(display_frame)
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

            # Alignment HUD
            cv2.putText(display_frame, f"Alignment - Frame {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"State: {state}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(display_frame, f"Confidence: {conf}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(display_frame, f"Error: {erro:.1f}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)
            cv2.putText(display_frame, f"Stable: {stable}/{ALIGN_STABLE}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            # Send a frame every 5 frames or when important
            if frame_count - last_frame_sent >= 5 or stable >= ALIGN_STABLE:
                send_frame_to_stream(display_frame)
                last_frame_sent = frame_count

            if conf==1 and abs(erro)<=ALIGN_TOL_PIX:
                stable+=1
                print(f"      Stable: {stable}/{ALIGN_STABLE} frames")
            else:
                stable=0

            if stable>=ALIGN_STABLE:
                print(f"      Alignment finished after {frame_count} frames!")
                # Final success frame
                cv2.putText(display_frame, "ALIGNMENT COMPLETE!", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                send_frame_to_stream(display_frame)

                _ = drive_cap(arduino, 70, 70, cap=ALIGN_CAP); time.sleep(0.10)
                _ = drive_cap(arduino, 0, 0); return True

            elapsed = time.time()-t0
            if elapsed > ALIGN_TIMEOUT:
                print(f"      Timeout after {elapsed:.1f}s ({frame_count} frames)")
                _ = drive_cap(arduino,0,0); return False

            raw.truncate(0); raw.seek(0)
    finally:
        raw.truncate(0)

def best_intersection_in_band(pts, h, band_y0, band_y1):
    """Choose the best intersection: try the band first, otherwise accept outside of it."""
    # First try to find one inside the main band
    cand_in_band = None
    best_y_in_band = -1

    cand_out_band = None
    best_y_out_band = h  # Start with the largest possible y (farthest)

    for (x,y) in pts:
        if band_y0 <= y <= band_y1:
            # Inside the band - choose the closest (largest y)
            if y > best_y_in_band:
                best_y_in_band = y
                cand_in_band = (x,y)
        else:
            # Outside the band - choose the closest (smallest y)
            if y < best_y_out_band:
                best_y_out_band = y
                cand_out_band = (x,y)

    # Prioritize intersections within the band, but accept outside when necessary
    return cand_in_band if cand_in_band is not None else cand_out_band

def go_to_next_intersection(arduino, camera, expected_node=None):
    """Reach the next intersection using the robust logic from robot_pedro.py.
    Returns (success: bool, obstacle_detected: bool)"""
    raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    last_err = 0.0
    lost_frames = 0
    # States: 'FOLLOW', 'LOST', 'APPROACHING', 'STOPPING', 'STOPPED'
    state = 'FOLLOW'
    obstacle_detected = False
    action_start_time = 0.0
    approach_start_time = 0.0
    last_known_y = -1.0  # Last valid Y position of the intersection
    last_intersection_point = None
    last_intersection_y = -1.0
    last_intersection_time = 0.0
    last_intersection_is_border = False
    intersection_descent_frames = 0
    intersection_descent_start_y = -1.0
    intersection_last_live_y = -1.0
    intersection_last_live_time = 0.0
    intersection_descent_confident = False
    planned_border = None if expected_node is None else is_border_node(expected_node)

    last_stop_candidate_is_border = bool(planned_border)
    current_stop_is_border = bool(planned_border)

    capture_iter = camera.capture_continuous(raw, format="bgr", use_video_port=True)
    try:
        while True:
            try:
                f = next(capture_iter)
            except PiCameraValueError as exc:
                print(f"   Warning: inconsistent camera buffer ({exc}). Resetting frame...")
                raw.truncate(0)
                raw.seek(0)
                continue
            img = f.array
            mask = build_binary_mask(img)
            img, erro, conf = processar_imagem(img)

            h, w = img.shape[:2]
            now = time.time()
            border_margin_px = int(w * BORDER_MARGIN_FRAC)

            intersections, detected_lines = detect_intersections(mask)
            intersections = list(intersections)  # Ensure we can sort/extend
            target_intersection = None
            target_y = -1.0
            is_border_intersection = False
            intersection_from_memory = False
            target_from_memory = False

            if intersections:
                filtered_intersections = [
                    inter for inter in intersections
                    if border_margin_px <= inter[0] <= w - border_margin_px
                ]

                if filtered_intersections:
                    filtered_intersections.sort(key=lambda p: p[1], reverse=True)
                    raw_target = filtered_intersections[0]
                    source_label = "FILTERED"
                else:
                    center_x = w // 2
                    intersections.sort(key=lambda p: (abs(p[0] - center_x), -p[1]))
                    raw_target = intersections[0]
                    source_label = "FALLBACK"

                target_intersection = (int(raw_target[0]), int(raw_target[1]))
                target_y = float(target_intersection[1])
                print(f"   Target intersection: {target_intersection} (x={target_intersection[0]}, y={target_y:.0f}) - {source_label}")
                is_border_intersection = (
                    target_intersection[0] <= border_margin_px or
                    target_intersection[0] >= w - border_margin_px
                )

                if planned_border is not None:
                    is_border_intersection = planned_border

            should_use_memory = (
                planned_border if planned_border is not None else (
                    is_border_intersection or last_intersection_is_border
                )
            )

            if (
                should_use_memory
                and target_intersection is not None
                and last_intersection_point is not None
                and last_intersection_y >= 0.0
            ):
                jump_threshold = max(INTERSECTION_DESCENT_TOL_PX * 2, h * INTERSECTION_REJECT_JUMP_FRAC)
                if (
                    intersection_descent_confident
                    and (last_intersection_y - target_y) > jump_threshold
                ):
                    print(
                        f"   Current intersection jumped too much (DeltaY={last_intersection_y - target_y:.0f} > {jump_threshold:.0f})."
                        " Keeping the previous memory."
                    )
                    target_intersection = None
                    target_y = -1.0

            extra_memory_valid = False
            if should_use_memory and intersection_descent_confident and intersection_last_live_time > 0.0:
                extra_memory_valid = (now - intersection_last_live_time) <= INTERSECTION_DESCENT_MEMORY_S

            memory_valid = (
                should_use_memory
                and target_intersection is None
                and last_intersection_point is not None
                and (
                    (now - last_intersection_time) <= INTERSECTION_MEMORY_S
                    or extra_memory_valid
                )
            )

            if memory_valid:
                dt = now - last_intersection_time
                projected_y = last_intersection_y + INTERSECTION_MEMORY_GROW_FRAC_PER_S * dt * h
                if intersection_last_live_y >= 0:
                    extra_cap_frac = (
                        BORDER_INTERSECTION_MEMORY_EXTRA_FRAC
                        if last_intersection_is_border
                        else INTERSECTION_MEMORY_EXTRA_FRAC
                    )
                    max_projected_y = intersection_last_live_y + extra_cap_frac * h
                    projected_y = min(projected_y, max_projected_y)
                projected_y = min(projected_y, float(h - 1))

                last_intersection_y = projected_y
                last_intersection_time = now
                last_intersection_point = (last_intersection_point[0], int(round(projected_y)))
                target_intersection = last_intersection_point
                target_y = projected_y
                is_border_intersection = last_intersection_is_border
                intersection_from_memory = True
                print(f"   Memorized intersection: {target_intersection} (border={is_border_intersection}) | proj_y={projected_y:.0f}")

            if target_intersection is not None:
                if should_use_memory and not intersection_from_memory:
                    if intersection_last_live_y >= 0 and target_y + INTERSECTION_DESCENT_TOL_PX >= intersection_last_live_y:
                        intersection_descent_frames += 1
                    else:
                        intersection_descent_frames = 1
                        intersection_descent_start_y = target_y

                    if intersection_descent_frames == 1:
                        intersection_descent_start_y = target_y

                    intersection_last_live_y = target_y
                    intersection_last_live_time = now

                    delta_y = target_y - intersection_descent_start_y
                    min_delta = h * INTERSECTION_DESCENT_MIN_DELTA_FRAC
                    intersection_descent_confident = (
                        intersection_descent_frames >= INTERSECTION_DESCENT_MIN_FRAMES
                        and delta_y >= min_delta
                        and target_y >= h * Y_START_SLOWING_FRAC
                    )

                last_intersection_point = target_intersection
                last_intersection_y = float(target_y)
                last_intersection_time = now
                last_intersection_is_border = is_border_intersection
                last_stop_candidate_is_border = is_border_intersection
            else:
                intersection_descent_confident = False
                intersection_descent_frames = 0
                intersection_descent_start_y = -1.0
                intersection_last_live_y = -1.0
                intersection_last_live_time = 0.0
                last_intersection_point = None
                last_intersection_y = -1.0
                last_intersection_is_border = False
                last_stop_candidate_is_border = False

            if not should_use_memory:
                intersection_descent_confident = False
                intersection_descent_frames = 0
                intersection_descent_start_y = -1.0
                intersection_last_live_y = -1.0
                intersection_last_live_time = 0.0

            if target_intersection is not None and target_intersection not in intersections:
                intersections = intersections + [target_intersection]

            intersection_source = "MEM" if intersection_from_memory else ("LIVE" if target_intersection is not None else "--")
            target_from_memory = intersection_from_memory

            y_start_frac = BORDER_Y_START_SLOWING_FRAC if is_border_intersection else Y_START_SLOWING_FRAC
            y_target_frac = BORDER_Y_TARGET_STOP_FRAC if is_border_intersection else Y_TARGET_STOP_FRAC
            Y_START_SLOWING = h * y_start_frac
            Y_TARGET_STOP = h * y_target_frac

            if is_border_intersection and intersection_last_live_y >= 0:
                pad_y = intersection_last_live_y + INTERSECTION_BORDER_STOP_PAD_FRAC * h
                min_stop = max(Y_START_SLOWING + 4.0, pad_y)
                Y_TARGET_STOP = min(Y_TARGET_STOP, min_stop)

            recent_intersection = (
                target_intersection is not None or
                (last_intersection_point is not None and (now - last_intersection_time) <= INTERSECTION_MEMORY_S)
            )

            if target_y != -1.0:
                print(f"   Intersection Y={target_y:.0f} [{intersection_source}] (border={is_border_intersection}) | target={Y_TARGET_STOP:.0f}")
                should_approach = target_y >= Y_START_SLOWING
                print(f"   Should approach: {should_approach} (Y >= {Y_START_SLOWING:.0f})")
            elif conf == 0:
                print(f"   Line lost! erro={erro}, conf={conf}")
            else:
                print(f"   Line OK: erro={erro:.1f}, conf={conf}")

            # --- Control State Machine (from robot_pedro.py) ---

            # 1. State transitions
            if state == 'FOLLOW':
                target_dbg = f"{target_y:.0f}" if target_y != -1.0 else "None"
                print(f"   State machine: conf={conf}, target_y={target_dbg} [{intersection_source}], Y_START_SLOWING={Y_START_SLOWING:.0f}")
                # Check if we should begin the approach (regardless of current confidence)
                if target_y != -1.0 and target_y >= Y_START_SLOWING:
                    print(f"   Intersection detected at Y={target_y:.0f}! Starting approach (Y_START_SLOWING={Y_START_SLOWING:.0f})")
                    state = 'APPROACHING'
                    approach_start_time = now
                    last_known_y = target_y
                    lost_frames = 0
                    current_stop_is_border = is_border_intersection
                elif conf == 0:
                    lost_frames += 1
                    if lost_frames >= LOST_MAX_FRAMES:
                        threshold_hit = (lost_frames == LOST_MAX_FRAMES)
                        pending_stop = last_known_y > Y_START_SLOWING
                        if not recent_intersection and not pending_stop:
                            if threshold_hit:
                                print("   Line lost (FOLLOW). Switching to LOST.")
                            state = 'LOST'
                            approach_start_time = 0.0
                            last_known_y = -1.0
                        else:
                            lost_frames = min(lost_frames, LOST_MAX_FRAMES)
                            if threshold_hit:
                                print("   Line missing, but recent intersection - staying in FOLLOW.")
                else:
                    lost_frames = 0
                    print(f"   Waiting to approach: target_y={target_y:.0f} <= Y_START_SLOWING={Y_START_SLOWING:.0f}")
                    last_err = erro
                    last_known_y = -1.0

            elif state == 'APPROACHING':
                # Check for line loss with tolerance
                if conf == 0:
                    lost_frames += 1
                    print(f"   Approaching, confidence lost! (Frame {lost_frames})")

                    if target_y != -1.0 and target_from_memory and is_border_intersection:
                        last_known_y = target_y
                        current_stop_is_border = True
                        if last_known_y >= Y_TARGET_STOP:
                            print("   Memory target reached. Crawling a bit more...")
                            state = 'STOPPING'
                            action_start_time = time.time()
                            approach_start_time = 0.0
                            last_known_y = -1.0
                            raw.truncate(0)
                            raw.seek(0)
                            continue

                    if lost_frames >= LOST_MAX_FRAMES:
                        threshold_hit = (lost_frames == LOST_MAX_FRAMES)
                        pending_stop = last_known_y > Y_START_SLOWING
                        if not recent_intersection and not pending_stop:
                            if threshold_hit:
                                print("   Line lost during approach. Switching to LOST.")
                            state = 'LOST'
                            approach_start_time = 0.0
                            last_known_y = -1.0
                        else:
                            lost_frames = min(lost_frames, LOST_MAX_FRAMES)
                            if threshold_hit:
                                print("   Approach: keeping state due to recent intersection/progress.")

                else:
                    lost_frames = 0

                    # Update the known intersection position
                    if target_y != -1.0:
                        last_known_y = target_y
                        current_stop_is_border = is_border_intersection

                        # Trigger 1: Did we reach the Y target?
                        if last_known_y >= Y_TARGET_STOP:
                            print("   Target (Y_TARGET_STOP) reached. Crawling a bit more...")
                            state = 'STOPPING'
                            action_start_time = time.time()
                            approach_start_time = 0.0
                            last_known_y = -1.0  # Reset for the next one
                            current_stop_is_border = is_border_intersection

                    # Trigger 2: Intersection disappeared completely (backup)
                    if target_y == -1.0 and last_known_y > Y_START_SLOWING:
                        print(f"   Intersection disappeared (was Y={last_known_y:.0f}). Stopping...")
                        state = 'STOPPING'
                        action_start_time = time.time()
                        approach_start_time = 0.0
                        last_known_y = -1.0  # Reset for the next one
                        current_stop_is_border = last_stop_candidate_is_border

                if state == 'APPROACHING' and approach_start_time > 0.0:
                    elapsed_approach = now - approach_start_time
                    if elapsed_approach > APPROACH_TIMEOUT_S:
                        print(f"   Approaching timeout ({elapsed_approach:.1f}s). Forcing stop.")
                        state = 'STOPPING'
                        action_start_time = now
                        approach_start_time = 0.0
                        last_known_y = -1.0
                        current_stop_is_border = last_stop_candidate_is_border

            elif state == 'STOPPING':
                crawl_limit = BORDER_CRAWL_DURATION_S if current_stop_is_border else CRAWL_DURATION_S
                if (time.time() - action_start_time) > crawl_limit:
                    print("   Full stop at the intersection!")
                    state = 'STOPPED'
                    current_stop_is_border = False

            elif state == 'LOST':
                if conf == 1:
                    print("   Line found again.")
                    state = 'FOLLOW'
                    lost_frames = 0
                    approach_start_time = 0.0
                    last_err = erro
                    last_known_y = -1.0

            # 2. State actions (set velocities)
            if state == 'FOLLOW':
                if conf == 1:
                    # Keep the base speed constant, only adjust with P
                    v_esq, v_dir = calcular_velocidades_auto(erro, VELOCIDADE_BASE)
                else:
                    # Tolerance window: keep going straight
                    v_esq, v_dir = VELOCIDADE_BASE, VELOCIDADE_BASE

            elif state == 'APPROACHING':
                if conf == 0:
                    base_speed = max(APPROACH_LOST_SPEED, APPROACH_FLOOR_SPEED)
                    v_esq, v_dir = calcular_velocidades_auto(0, base_speed)
                else:
                    progress = 0.0
                    denom = (Y_TARGET_STOP - Y_START_SLOWING)
                    if denom > 1e-6:
                        progress = (last_known_y - Y_START_SLOWING) / denom
                    progress = float(np.clip(progress, 0.0, 1.0))
                    eased = progress ** 1.35
                    target_speed = VELOCIDADE_BASE - (VELOCIDADE_BASE - APPROACH_FLOOR_SPEED) * eased
                    base_speed = int(np.clip(target_speed, APPROACH_FLOOR_SPEED, VELOCIDADE_MAX))
                    v_esq, v_dir = calcular_velocidades_auto(erro, base_speed)

            elif state == 'STOPPING':
                # "Move a little more" - straight crawl
                v_esq, v_dir = CRAWL_SPEED, CRAWL_SPEED

            elif state == 'STOPPED':
                v_esq, v_dir = 0, 0

            elif state == 'LOST':
                # Search logic
                turn = SEARCH_SPEED if last_err >= 0 else -SEARCH_SPEED
                v_esq, v_dir = int(turn), int(-turn)

            obs = drive_cap(arduino, v_esq, v_dir, cap=ALIGN_CAP)
            if obs:
                obstacle_detected = True
                print("!!! OBSTACLE DETECTED during navigation !!!")
                drive_cap(arduino, 0, 0)
                return False, True  # Failed due to obstacle

            # ---------------- VISUALIZATION ----------------
            display_frame = img.copy()
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
            display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

            # Draw lines (green)
            for rho, theta in detected_lines:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1 = int(x0 + 1000 * (-b));  y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b));  y2 = int(y0 - 1000 * (a))
                cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Intersections (red)
            for idx, (x, y) in enumerate(intersections, 1):
                cv2.circle(display_frame, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(display_frame, (x, y), 12, (255, 255, 255), 2)
                cv2.putText(display_frame, f"{idx}", (x + 15, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Highlight the target intersection
            if target_intersection is not None:
                cv2.circle(display_frame, target_intersection, 15, (255, 0, 255), 3)

            # HUD with the current state
            state_color = (0, 255, 0)  # Green for FOLLOW
            if state == 'LOST': state_color = (0, 0, 255)  # Red
            elif state == 'APPROACHING': state_color = (0, 255, 255)  # Yellow
            elif state == 'STOPPING': state_color = (255, 0, 255)  # Magenta
            elif state == 'STOPPED': state_color = (255, 0, 0)  # Blue

            approaching_border = (
                current_stop_is_border
                or (planned_border is True)
                or (planned_border is None and is_border_intersection)
            )
            hud_state = "APPROACHING B." if (state == 'APPROACHING' and approaching_border) else state

            cv2.putText(display_frame, f"State: {hud_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            cv2.putText(display_frame, f"Conf: {conf}  Lost: {lost_frames}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)
            cv2.putText(display_frame, f"Y_target: {target_y:.0f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            send_frame_to_stream(display_frame)

            if state == 'STOPPED':
                return True, False  # Success, no obstacle

            raw.truncate(0); raw.seek(0)

            # Safety timeout
            if (time.time() - time.time()) > 20.0:
                print("   Intersection detection timeout")
                drive_cap(arduino, 0, 0)
                return False, False  # Failed, no obstacle

    finally:
        try:
            capture_iter.close()
        except Exception:
            pass
        raw.truncate(0)

# ====================== Planning and Execution ======================
def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def front_left_right_corners(sx, sy, orient):
    """Map the current square to the two intersections reachable by the pivot."""

    # For each orientation we consider the two corners "ahead" of the robot.
    # The first tuple element represents turning left; the second, turning right.
    if orient == 0:   # North
        return ((sx, sy), (sx, sy + 1))
    if orient == 1:   # East
        return ((sx + 1, sy), (sx + 1, sy + 1))
    if orient == 2:   # South
        return ((sx + 1, sy + 1), (sx + 1, sy))
    if orient == 3:   # West
        return ((sx + 1, sy), (sx, sy))
    raise ValueError

def get_accessible_intersections(sx, sy, orient):
    """Return all intersections accessible from a square in a given orientation."""
    left_corner, right_corner = front_left_right_corners(sx, sy, orient)
    return [left_corner, right_corner]

def choose_best_pivot_intersection(sx, sy, cur_dir, target, grid=GRID_NODES):
    """Select the accessible intersection that yields the best A* path to the target.

    Returns (intersection, path) or None if nothing accessible produces a path.
    """
    accessible = get_accessible_intersections(sx, sy, cur_dir)

    best_choice = None
    best_path = None
    best_len = None
    best_manhattan = None

    for candidate in accessible:
        cx, cy = candidate
        if not (0 <= cx < grid[0] and 0 <= cy < grid[1]):
            continue

        path = a_star(candidate, target, grid)
        if path is None:
            continue

        path_len = len(path)
        dist = manhattan(candidate, target)

        if best_choice is None:
            best_choice = candidate
            best_path = path
            best_len = path_len
            best_manhattan = dist
            continue

        if path_len < best_len:
            best_choice = candidate
            best_path = path
            best_len = path_len
            best_manhattan = dist
            continue

        if path_len == best_len and dist < best_manhattan:
            best_choice = candidate
            best_path = path
            best_manhattan = dist

    if best_choice is None:
        return None

    return best_choice, best_path

def a_star(start,goal,grid=(5,5)):
    open_set={start}; came={}; g={start:0}; f={start:manhattan(start,goal)}
    W,H=grid
    def neigh(n):
        x,y=n
        cand=[]
        # Check all 4 directions and filter blocked edges
        neighbors = []
        if y-1>=0: neighbors.append((x,y-1))
        if x+1<W : neighbors.append((x+1,y))
        if y+1<H: neighbors.append((x,y+1))
        if x-1>=0: neighbors.append((x-1,y))
        
        # Filter out blocked edges
        for neighbor in neighbors:
            if not is_edge_blocked(n, neighbor):
                cand.append(neighbor)
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
    # Coordinates (row, column) - row increases downward, column increases to the right
    # North: row decreases (b[0] < a[0])
    # South: row increases (b[0] > a[0])
    # East: column increases (b[1] > a[1])
    # West: column decreases (b[1] < a[1])
    if b[0] < a[0]: return 0  # North
    if b[0] > a[0]: return 2  # South
    if b[1] > a[1]: return 1  # East
    return 3  # West
def relative_turn(cur_dir,want_dir): return {0:'F',1:'R',2:'U',3:'L'}[(want_dir-cur_dir)%4]

def dir_name(d):
    return {0:'North', 1:'East', 2:'South', 3:'West'}[d]

def leave_square_to_best_corner(arduino, camera, sx, sy, cur_dir, target, target_intersection=None, return_arrival_dir=True):
    """Exit the square using the declared orientation (assumed correct).
    path: full A* path used to reach a specific intersection if available.
    """
    dir_text = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}[cur_dir]
    print(f"Walking out of square ({sx},{sy})")
    print(f"   Orientation: {dir_text}")
    print(f"   Destination: {target}")

    left_corner, right_corner = front_left_right_corners(sx, sy, cur_dir)

    # If we have a specific A* target intersection, use it directly when possible
    if target_intersection is not None:
        if target_intersection == left_corner:
            side_hint = 'L'
            chosen = left_corner
        elif target_intersection == right_corner:
            side_hint = 'R'
            chosen = right_corner
        else:
            # Target intersection not directly accessible, fallback based on the final target
            dl = manhattan(left_corner, target)
            dr = manhattan(right_corner, target)
            side_hint = 'L' if dl <= dr else 'R'
            chosen = left_corner if side_hint=='L' else right_corner
            print(f"   Warning: target intersection {target_intersection} not accessible, using fallback")
    else:
        # Fallback to the original logic
        dl = manhattan(left_corner, target)
        dr = manhattan(right_corner, target)
        side_hint = 'L' if dl <= dr else 'R'
        chosen = left_corner if side_hint=='L' else right_corner

    turn_desc = {'L':'left', 'R':'right'}[side_hint]
    print(f"   Choosing corner {chosen} (turn: {turn_desc})")
    if target_intersection is None:
        print(f"   Manhattan distance: {dl} vs {dr}")

    # Blind straight
    if not straight_until_seen_then_lost(arduino, camera):
        print("Initial straight failed.")
        return None, None, False

    # Pivot: turn until the line is visible
    if not spin_in_place_until_seen(arduino, camera, side_hint=side_hint, orient=cur_dir):
        print("Pivot failed (line not seen).")
        return None, None, False

    # Quick alignment: center the line for a few frames
    print("   Centering the line...")
    raw_temp = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
    align_count = 0
    try:
        for f in camera.capture_continuous(raw_temp, format="bgr", use_video_port=True):
            if align_count >= 10:  # Maximum of 10 alignment frames
                break
            img_temp = f.array
            _, erro, conf = processar_imagem(img_temp)

            if conf == 1 and abs(erro) <= 10:  # Already well centered
                print(f"   Line aligned (error: {erro:.1f})")
                break

            # Simple correction movement
            if conf == 1:
                base_speed = 70
                if erro > 5:  # Line on the right, turn left
                    _ = drive_cap(arduino, base_speed-20, base_speed+20, cap=ALIGN_CAP)
                elif erro < -5:  # Line on the left, turn right
                    _ = drive_cap(arduino, base_speed+20, base_speed-20, cap=ALIGN_CAP)
                else:  # Centered
                    _ = drive_cap(arduino, base_speed, base_speed, cap=ALIGN_CAP)
            else:
                _ = drive_cap(arduino, 60, 60, cap=ALIGN_CAP)  # Move slowly if the line is lost

            time.sleep(0.05)  # Frame rate control
            align_count += 1
            raw_temp.truncate(0)
    finally:
        _ = drive_cap(arduino, 0, 0); time.sleep(0.1)  # Stop before continuing, ignore obstacle during initial alignment
        raw_temp.truncate(0)

    # Head toward the first intersection
    success, obstacle = go_to_next_intersection(arduino, camera, expected_node=chosen)
    if not success:
        if obstacle:
            print("Failed to reach the intersection - OBSTACLE DETECTED.")
            return None, None, False, True  # Signal obstacle
        else:
            print("Failed to reach the intersection.")
            return None, None, False, False

    # The pivot turned the robot toward the chosen corner
    # Update cur_dir based on side_hint
    if side_hint == 'L':
        cur_dir = (cur_dir - 1) % 4  # Turned left
    elif side_hint == 'R':
        cur_dir = (cur_dir + 1) % 4  # Turned right

    print(f"Pivot complete - now facing {dir_name(cur_dir)}")

    if return_arrival_dir:
        # The robot reaches the intersection coming from the current direction
        arrival_dir = cur_dir
        print(f"Arriving at intersection {chosen} from {dir_name(arrival_dir)}")
        return chosen, cur_dir, True, arrival_dir, False  # No obstacle
    else:
        return chosen, cur_dir, True, False  # No obstacle

# exec_turn removed - actions executed directly in follow_path using robot_pedro.py logic

def follow_path(arduino, start_node, start_dir, path, camera, arrival_dir=None, target=None):
    """
    The robot is already at the first intersection (start_node) after leave_square_to_best_corner.
    arrival_dir: arrival direction at the first intersection (0=N, 1=E, 2=S, 3=W).
    If None, assume arrival_dir = start_dir.
    target: final destination node for rerouting in case of obstacles.
    Returns (final_node, final_dir, success)
    """
    cur_node=start_node
    # If not specified, assume it arrived facing start_dir
    actual_arrival_dir = arrival_dir if arrival_dir is not None else start_dir
    cur_dir = actual_arrival_dir  # Start with the arrival direction
    
    if target is None:
        target = path[-1]  # Default to last node in path

    print(f"Arriving at the first intersection {start_node} from {dir_name(actual_arrival_dir)}")

    _ = drive_cap(arduino,0,0); time.sleep(0.1)

    # Show the full path
    print("A* PATH CALCULATED:")
    path_str = " -> ".join([f"({x},{y})" for x,y in path])
    print(f"   {path_str}")
    print(f"   Total: {len(path)-1} moves")
    print()

    # Check if we are already at the destination
    if start_node == path[-1]:
        print(f"Already at the destination ({start_node[0]},{start_node[1]})!")
        return cur_node,cur_dir,True

    # The robot is already at the first intersection after leave_square_to_best_corner
    # No need to move anywhere yet, just confirm our position
    print(f"Already at the first intersection {start_node}")
    cur_node = start_node
    print()

    # Execute each path step (starting from the second node)
    for i in range(1,len(path)):
        nxt=path[i]
        want=orientation_of_step(cur_node, nxt)
        rel=relative_turn(cur_dir,want)

        # Debug: show calculations
        print(f"   DEBUG: cur_node={cur_node}, nxt={nxt}, cur_dir={cur_dir}({dir_name(cur_dir)}), want={want}({dir_name(want)}), rel={rel}")

        # Show each specific turn
        turn_names = {'F':'straight', 'L':'left', 'R':'right', 'U':'u-turn'}
        print(f"Intersection ({cur_node[0]},{cur_node[1]}): turn {turn_names[rel]} toward ({nxt[0]},{nxt[1]})")
        print(f"   Current dir={cur_dir}, want={want}, rel={rel}")

        # IMPORTANT: come to a full stop before turning
        drive_cap(arduino, 0, 0); time.sleep(0.3)
        print(f"   Stopped to execute turn")

        # Execute the action based on the relative turn (robot_pedro.py logic)
        post_turn_settle_s = 0.0

        if rel == 'F':
            # GO_STRAIGHT: Already facing the correct direction, just update heading
            print("   Already facing the correct direction, moving forward...")
            cur_dir = want

        elif rel == 'L':
            # TURN_LEFT: Turn 90 degrees left
            print(f"   Turning left: drive_cap({arduino}, {-TURN_SPEED}, {TURN_SPEED}) for {TURN_DURATION_S}s")
            drive_cap(arduino, -TURN_SPEED, TURN_SPEED)
            time.sleep(TURN_DURATION_S)
            print(f"   Stopping left turn...")
            drive_cap(arduino, 0, 0); time.sleep(0.3)
            print("   Left turn complete")
            cur_dir = want

        elif rel == 'R':
            # TURN_RIGHT: Turn 90 degrees right
            print(f"   Turning right: drive_cap({arduino}, {TURN_SPEED}, {-TURN_SPEED}) for {TURN_DURATION_S}s")
            drive_cap(arduino, TURN_SPEED, -TURN_SPEED)
            time.sleep(TURN_DURATION_S)
            print(f"   Stopping right turn...")
            drive_cap(arduino, 0, 0); time.sleep(0.3)
            print("   Right turn complete")
            cur_dir = want

        elif rel == 'U':
            # U-turn: 180 degrees
            print("   Performing a u-turn...")
            drive_cap(arduino, TURN_SPEED, -TURN_SPEED)
            time.sleep(UTURN_DURATION_S)
            drive_cap(arduino, 0, 0); time.sleep(0.4)
            print("   U-turn complete")
            cur_dir = want
            post_turn_settle_s = 1.0

        print(f"   Action executed")

        if post_turn_settle_s > 0:
            print(f"   Waiting {post_turn_settle_s:.1f}s to stabilize after the u-turn...")
            time.sleep(post_turn_settle_s)

        # Now go to the next intersection following the line
        success, obstacle = go_to_next_intersection(arduino, camera, expected_node=nxt)
        if not success:
            if obstacle:
                print(f"   OBSTACLE detected while going to ({nxt[0]},{nxt[1]})")
                
                # Add blocked edge
                add_blocked_edge(cur_node, nxt)
                
                # Execute U-turn
                handle_obstacle_uturn(arduino, camera)
                
                # Recalculate path avoiding the blocked edge
                print(f"   Recalculating path from {cur_node} to {target}")
                new_path = a_star(cur_node, target, GRID_NODES)
                
                if new_path is None:
                    print("   ERROR: No alternative path found!")
                    return cur_node, cur_dir, False
                
                print(f"   NEW PATH: {' -> '.join([f'({x},{y})' for x,y in new_path])}")
                
                # Continue with new path (already at cur_node)
                # Recursively call follow_path with the new route
                return follow_path(arduino, cur_node, cur_dir, new_path, camera, cur_dir, target)
            else:
                print(f"   Failed to reach ({nxt[0]},{nxt[1]})")
                return cur_node, cur_dir, False

        # After the movement, the robot keeps the desired direction
        print(f"   Arrived at ({nxt[0]},{nxt[1]})")
        print()
        cur_node=nxt
        cur_dir = want  # Maintain the direction we were heading

    print(f"Reached the final destination!")
    return cur_node,cur_dir,True

# =================================== REMOTE CONTROL ===================================

# =================================== MAIN ===================================
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--square', type=int, nargs=2, required=True, metavar=('SX','SY'))
    ap.add_argument('--orient', type=str, required=True, help='N/E/S/W')
    ap.add_argument('--target', type=int, nargs=2, required=True, metavar=('TX','TY'))
    ap.add_argument('--no-return', action='store_true')
    return ap.parse_args()

# Global variables for streaming
stream_socket = None
stream_context = None

# Initial detection control
initial_frames_ignored = 0
IGNORE_INITIAL_FRAMES = 15  # Ignore the first 15 frames to avoid floor detection

def init_streaming():
    """Initialize the ZMQ socket for streaming."""
    global stream_socket, stream_context
    if stream_socket is None:
        stream_context = zmq.Context()
        stream_socket = stream_context.socket(zmq.PUB)
        stream_socket.bind('tcp://*:5555')
        print("Streaming ZMQ initialized at tcp://*:5555")

def send_frame_to_stream(display_frame):
    """Send a specific frame to the ZMQ stream."""
    global stream_socket
    if stream_socket:
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        stream_socket.send(base64.b64encode(buffer))

def send_basic_frame(camera, message="Processing..."):
    """Send a basic camera frame with a message."""
    try:
        raw = PiRGBArray(camera, size=(IMG_WIDTH, IMG_HEIGHT))
        camera.capture(raw, format="bgr", use_video_port=True)
        img = raw.array
        mask = build_binary_mask(img)

        # Basic visualization
        display_frame = img.copy()
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        display_frame = cv2.addWeighted(display_frame, 0.7, mask_color, 0.3, 0)

        # Basic HUD
        cv2.putText(display_frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        send_frame_to_stream(display_frame)
    except Exception as e:
        print(f"Error sending basic frame: {e}")

def main():
    args=parse_args()
    sx,sy=args.square; tx,ty=args.target
    orient=args.orient.strip().upper()
    if orient not in ('N','E','S','W','O'): raise SystemExit("orient must be N/E/S/W")
    cur_dir={'N':0,'E':1,'S':2,'W':3,'O':3}[orient]
    if not (0<=sx<=3 and 0<=sy<=3): raise SystemExit("square 0..3 0..3")
    if not (0<=tx<=4 and 0<=ty<=4): raise SystemExit("target 0..4 0..4")
    target=(tx,ty)


    # Initialize camera and streaming
    camera = PiCamera(); camera.resolution=(IMG_WIDTH, IMG_HEIGHT); camera.framerate=24
    time.sleep(1.0)  # longer warm-up
    init_streaming()  # Initialize ZMQ

    # Wait a bit longer before the first frame
    time.sleep(0.5)

    # Send the first basic frame with retries
    for attempt in range(3):
        try:
            send_basic_frame(camera, "System initialized - waiting for command")
            print("First frame sent successfully")
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(0.5)
    else:
        print("Could not send first frame, continuing...")

    arduino = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=0.01); time.sleep(2)
    try:
        arduino.write(b'A10')
        try: print("Arduino:", arduino.readline().decode('utf-8').strip())
        except Exception: pass
    except Exception: pass

    # Robot states
    manual_mode = False
    last_key = None
    auto_state = "INIT"  # States: INIT, LEAVING, NAVIGATING, RETURNING, DONE

    try:
        print(f"START: Square ({sx},{sy})")
        print(f"DESTINATION: Node ({tx},{ty})")
        print("AUTONOMOUS MODE")
        print()

        # Determine the initial intersection using the shortest viable path
        best_choice = choose_best_pivot_intersection(sx, sy, cur_dir, (tx, ty), GRID_NODES)
        accessible = get_accessible_intersections(sx, sy, cur_dir)

        if best_choice is None:
            print("Warning: No accessible path via refined choice; using simple heuristic.")
            start_intersection = min(accessible, key=lambda inter: manhattan(inter, (tx, ty)))
            chosen_path = a_star(start_intersection, (tx, ty), GRID_NODES)
        else:
            start_intersection, chosen_path = best_choice

        if chosen_path is None:
            print("No path found by A*.")
            send_basic_frame(camera, "ERROR: Path not found!")
            return

        print(f"Initial intersection chosen: {start_intersection} (based on orientation and target)")

        # Calculate A* from the initial intersection to the destination
        print("RUNNING A* TO CALCULATE PATH...")
        send_basic_frame(camera, "Calculating A* path...")

        print(f"PATH: {' -> '.join([f'({x},{y})' for x,y in chosen_path])}")
        send_basic_frame(camera, f"Path: {' -> '.join([f'({x},{y})' for x,y in chosen_path])}")

        # For the initial pivot, go to the chosen intersection exactly
        target_intersection = start_intersection
        print(f"Best accessible intersection: {target_intersection} (based on orientation)")

        # Execute autonomous navigation logic
        send_basic_frame(camera, f"Square ({sx},{sy}) -> Node ({tx},{ty})")

        print("Running leave_square_to_best_corner...")
        result = leave_square_to_best_corner(arduino, camera, sx, sy, cur_dir, target, target_intersection)
        print(f"leave_square_to_best_corner returned: {result}")
        
        # Handle different return formats
        if len(result) == 5:
            start_node, cur_dir, ok, arrival_dir, obstacle = result
        elif len(result) == 4:
            start_node, cur_dir, ok, potential_arrival_or_obstacle = result
            # Check if it's arrival_dir (int) or obstacle (bool)
            if isinstance(potential_arrival_or_obstacle, bool):
                obstacle = potential_arrival_or_obstacle
                arrival_dir = cur_dir
            else:
                arrival_dir = potential_arrival_or_obstacle
                obstacle = False
        else:
            start_node, cur_dir, ok = result
            arrival_dir = cur_dir
            obstacle = False
            
        if not ok:
            if obstacle:
                print("Failed during exit - OBSTACLE DETECTED.")
                # Could try alternative route from start here
                send_basic_frame(camera, "ERROR: Obstacle in initial path!")
            else:
                print("Failed during exit.")
                send_basic_frame(camera, "ERROR: Exit failed")
            return

        # Calculate path from the chosen intersection to the destination
        print(f"Calculating path from intersection {start_node} to target {target}")
        if chosen_path and chosen_path[0] == start_node:
            optimized_path = chosen_path
        else:
            optimized_path = a_star(start_node, target, GRID_NODES)
        if optimized_path is None:
            print("No path found from the chosen intersection.")
            send_basic_frame(camera, "ERROR: Path not found!")
            return

        print(f"PATH: {' -> '.join([f'({x},{y})' for x,y in optimized_path])}")
        send_basic_frame(camera, f"Navigating: {' -> '.join([f'({x},{y})' for x,y in optimized_path])}")

        _, cur_dir, ok = follow_path(arduino, start_node, cur_dir, optimized_path, camera, arrival_dir, target)
        if not ok:
            print("Navigation failed.")
            send_basic_frame(camera, "ERROR: Navigation failed")
            return
        print("Delivery completed successfully!")
        send_basic_frame(camera, "Delivery completed!")
        celebrate_delivery(arduino)

        if not args.no_return:
            print("CALCULATING RETURN PATH...")
            send_basic_frame(camera, "Calculating return path...")

            back_path = a_star(target, (sx, sy), GRID_NODES)
            if back_path is None:
                print("No return path found.")
                send_basic_frame(camera, "ERROR: Return path not found")
                return

            print(f"RETURN PATH: {' -> '.join([f'({x},{y})' for x,y in back_path])}")
            send_basic_frame(camera, f"Return: {' -> '.join([f'({x},{y})' for x,y in back_path])}")

            # For the return trip, assume we arrived facing cur_dir
            start_square = (sx, sy)
            _, _, ok = follow_path(arduino, target, cur_dir, back_path, camera, cur_dir, start_square)
            if not ok:
                print("Return failed.")
                send_basic_frame(camera, "ERROR: Return failed")
                return
            print("Return completed successfully!")
            send_basic_frame(camera, "Return completed!")

        print("MISSION COMPLETE!")
        send_basic_frame(camera, "MISSION COMPLETE")
        time.sleep(3.0)

    except Exception as e:
        print(f"Error during execution: {e}")
        try:
            enviar_comando_motor_serial(arduino, 0, 0)
            arduino.write(b'a'); arduino.close()
        except Exception: pass
        camera.close()
        return

    finally:
        try:
            enviar_comando_motor_serial(arduino, 0, 0)
            arduino.write(b'a'); arduino.close()
        except Exception: pass
        camera.close()

if __name__=='__main__':
    main()
