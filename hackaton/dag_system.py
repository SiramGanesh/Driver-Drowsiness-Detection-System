import cv2
import mediapipe as mp
import numpy as np
import pygame
from scipy.spatial import distance as dist
import time
import os

# ======================================
# DRIVER ATTENTION GUARDIAN (DAG)
# ACTUAL FINAL FINAL HACKATHON VERSION
# ======================================

# -------------------------------
# INIT SOUND (FIXED)
# -------------------------------
ALARM_FILE = "alert.wav"

alarm_loaded = False
alarm_on = False
alarm_sound = None
alarm_channel = None

try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    alarm_sound = pygame.mixer.Sound(ALARM_FILE)
    alarm_loaded = True
    print("✅ alert.wav loaded successfully")
except Exception as e:
    print("⚠️ Sound init failed:", e)
    print("⚠️ Make sure alert.wav is in the same folder.")

# -------------------------------
# CREATE ALERT FOLDER
# -------------------------------
if not os.path.exists("alerts"):
    os.makedirs("alerts")

# -------------------------------
# MEDIAPIPE FACE MESH
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# FACIAL LANDMARKS
# -------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454
FOREHEAD = 10
CHIN = 152

# -------------------------------
# THRESHOLDS
# -------------------------------
EAR_THRESHOLD = 0.23
DROWSY_FRAMES = 20

HORIZONTAL_DISTRACT_THRESHOLD = 30
UP_DISTRACT_THRESHOLD = 9
DOWN_DISTRACT_THRESHOLD = 18

DISTRACTION_TIME_THRESHOLD = 2.5  # seconds

# ------------------------------
# STATE VARIABLES
# -------------------------------
drowsy_counter = 0
risk_score = 10

last_alert_time = 0
alert_cooldown = 5
event_log = []

# Distraction timing
distraction_start_time = None
distraction_duration = 0

# -------------------------------
# FUNCTIONS
# -------------------------------
def play_alarm():
    global alarm_on, alarm_channel
    if alarm_loaded and not alarm_on:
        try:
            alarm_channel = alarm_sound.play(loops=-1)
            alarm_on = True
        except Exception as e:
            print("⚠️ Could not play alarm:", e)

def stop_alarm():
    global alarm_on, alarm_channel
    if alarm_loaded and alarm_on:
        try:
            if alarm_channel is not None:
                alarm_channel.stop()
            alarm_on = False
        except Exception as e:
            print("⚠️ Could not stop alarm:", e)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_eye_points(landmarks, eye_indices, w, h):
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))
    return np.array(points, dtype=np.int32)

def log_event(event_text):
    current_time = time.strftime("%H:%M:%S")

    # limit message length
    max_chars = 42
    if len(event_text) > max_chars:
        event_text = event_text[:max_chars] + "..."

    entry = f"[{current_time}] {event_text}"

    if len(event_log) == 0 or event_log[-1] != entry:
        event_log.append(entry)

    if len(event_log) > 6:
        event_log.pop(0)

def save_alert_image(frame, event_type):
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time > alert_cooldown:
        filename = f"alerts/{event_type}_{int(current_time)}.jpg"
        cv2.imwrite(filename, frame)
        last_alert_time = current_time
        log_event(f"{event_type.upper()} snapshot saved")

def draw_transparent_box(frame, x1, y1, x2, y2, color, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def get_distraction_direction(horizontal_offset, vertical_offset):
    if horizontal_offset >= HORIZONTAL_DISTRACT_THRESHOLD and vertical_offset >= DOWN_DISTRACT_THRESHOLD:
        return "DOWN-RIGHT"
    elif horizontal_offset <= -HORIZONTAL_DISTRACT_THRESHOLD and vertical_offset >= DOWN_DISTRACT_THRESHOLD:
        return "DOWN-LEFT"
    elif horizontal_offset >= HORIZONTAL_DISTRACT_THRESHOLD and vertical_offset <= -UP_DISTRACT_THRESHOLD:
        return "UP-RIGHT"
    elif horizontal_offset <= -HORIZONTAL_DISTRACT_THRESHOLD and vertical_offset <= -UP_DISTRACT_THRESHOLD:
        return "UP-LEFT"
    elif horizontal_offset >= HORIZONTAL_DISTRACT_THRESHOLD:
        return "RIGHT"
    elif horizontal_offset <= -HORIZONTAL_DISTRACT_THRESHOLD:
        return "LEFT"
    elif vertical_offset >= DOWN_DISTRACT_THRESHOLD:
        return "DOWN"
    elif vertical_offset <= -UP_DISTRACT_THRESHOLD:
        return "UP"
    return "CENTER"

def draw_ui(frame, status, ear, horizontal_offset, vertical_offset,
            risk_score, fps, distraction_direction, distraction_duration):
    h, w = frame.shape[:2]

    # -------------------------------
    # TOP HEADER
    # -------------------------------
    draw_transparent_box(frame, 0, 0, w, 42, (20, 20, 20), 0.85)
    cv2.putText(frame, "DRIVER ATTENTION GUARDIAN (DAG)", (18, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1)

    cv2.putText(frame, f"FPS: {int(fps)}", (w - 95, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # -------------------------------
    # LEFT DASHBOARD PANEL
    # -------------------------------
    draw_transparent_box(frame, 15, 55, 330, 275, (30, 30, 30), 0.75)

    cv2.putText(frame, "SYSTEM DASHBOARD", (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    cv2.putText(frame, f"Status      : {status}", (25, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

    cv2.putText(frame, f"EAR Value   : {ear:.2f}", (25, 133),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)

    cv2.putText(frame, f"X Offset    : {horizontal_offset}", (25, 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)

    cv2.putText(frame, f"Y Offset    : {vertical_offset}", (25, 183),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)

    cv2.putText(frame, f"Direction   : {distraction_direction}", (25, 208),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)

    cv2.putText(frame, f"Away Time   : {distraction_duration:.1f}s", (25, 233),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)

    cv2.putText(frame, f"Risk Score  : {risk_score}%", (25, 258),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    # -------------------------------
    # RISK BAR
    # -------------------------------
    cv2.rectangle(frame, (165, 245), (300, 258), (60, 60, 60), -1)
    bar_width = int((risk_score / 100) * 135)

    if risk_score < 40:
        bar_color = (0, 255, 0)
    elif risk_score < 75:
        bar_color = (0, 255, 255)
    else:
        bar_color = (0, 0, 255)

    cv2.rectangle(frame, (165, 245), (165 + bar_width, 258), bar_color, -1)

    # -------------------------------
    # ALERT BOX
    # -------------------------------
    if status == "DROWSY":
        draw_transparent_box(frame, 360, 60, 710, 115, (0, 0, 255), 0.85)
        cv2.putText(frame, "DROWSINESS ALERT!", (400, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

    elif status == "DISTRACTED":
        draw_transparent_box(frame, 360, 60, 850, 115, (0, 165, 255), 0.85)
        cv2.putText(frame, f"DISTRACTION ALERT! ({distraction_direction})", (375, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    elif status == "NO FACE":
        draw_transparent_box(frame, 390, 60, 690, 115, (255, 0, 0), 0.85)
        cv2.putText(frame, "NO DRIVER FACE!", (435, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    # -------------------------------
    # EVENT LOG PANEL
    # -------------------------------
    draw_transparent_box(frame, 15, h - 135, 470, h - 15, (30, 30, 30), 0.75)
    cv2.putText(frame, "RECENT EVENTS", (25, h - 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    y = h - 83
    for event in event_log:
        safe_event = event[:52] + "..." if len(event) > 52 else event
        cv2.putText(frame, safe_event, (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1)
        y += 18

# -------------------------------
# CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

# Fullscreen window setup
cv2.namedWindow("Driver Attention Guardian (DAG)", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Driver Attention Guardian (DAG)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("==============================================")
print(" DRIVER ATTENTION GUARDIAN (DAG)")
print(" AI-Powered Driver Monitoring System")
print("==============================================")
print("Press 'q' to quit")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_status = "AWAKE"
    ear = 0.0
    horizontal_offset = 0
    vertical_offset = 0
    risk_score = 10
    distraction_direction = "CENTER"

    drowsy_detected = False
    distracted_detected = False
    no_face_detected = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Eye points
            left_eye = get_eye_points(landmarks, LEFT_EYE, w, h)
            right_eye = get_eye_points(landmarks, RIGHT_EYE, w, h)

            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

            # EAR
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Drowsiness
            if ear < EAR_THRESHOLD:
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            if drowsy_counter >= DROWSY_FRAMES:
                drowsy_detected = True

            # Distraction
            nose_x = int(landmarks[NOSE_TIP].x * w)
            nose_y = int(landmarks[NOSE_TIP].y * h)

            left_x = int(landmarks[LEFT_FACE].x * w)
            right_x = int(landmarks[RIGHT_FACE].x * w)
            forehead_y = int(landmarks[FOREHEAD].y * h)
            chin_y = int(landmarks[CHIN].y * h)

            face_center_x = (left_x + right_x) // 2
            face_center_y = (forehead_y + chin_y) // 2

            horizontal_offset = nose_x - face_center_x
            vertical_offset = nose_y - face_center_y

            distraction_direction = get_distraction_direction(horizontal_offset, vertical_offset)

            # Guide points
            cv2.circle(frame, (nose_x, nose_y), 4, (255, 0, 0), -1)
            cv2.circle(frame, (face_center_x, face_center_y), 4, (0, 255, 255), -1)
            cv2.line(frame, (face_center_x, face_center_y), (nose_x, nose_y), (255, 255, 0), 2)

            # Distraction timer
            currently_distracted = distraction_direction != "CENTER"

            if currently_distracted:
                if distraction_start_time is None:
                    distraction_start_time = time.time()

                distraction_duration = time.time() - distraction_start_time

                if distraction_duration >= DISTRACTION_TIME_THRESHOLD:
                    distracted_detected = True
            else:
                distraction_start_time = None
                distraction_duration = 0

    else:
        no_face_detected = True
        drowsy_counter = 0
        distraction_start_time = None
        distraction_duration = 0

    # Priority logic
    if drowsy_detected:
        current_status = "DROWSY"
        risk_score = 90
        play_alarm()
        save_alert_image(frame, "drowsy")
        log_event("DROWSINESS detected")

    elif distracted_detected:
        current_status = "DISTRACTED"
        risk_score = 75
        play_alarm()
        save_alert_image(frame, f"distracted_{distraction_direction.lower().replace('-', '_')}")
        log_event(f"DISTRACTION detected ({distraction_direction})")

    elif no_face_detected:
        current_status = "NO FACE"
        risk_score = 35
        stop_alarm()
        log_event("No face detected")

    else:
        current_status = "AWAKE"
        risk_score = 10
        stop_alarm()

    draw_ui(
        frame,
        current_status,
        ear,
        horizontal_offset,
        vertical_offset,
        risk_score,
        fps,
        distraction_direction,
        distraction_duration
    )

    cv2.imshow("Driver Attention Guardian (DAG)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()