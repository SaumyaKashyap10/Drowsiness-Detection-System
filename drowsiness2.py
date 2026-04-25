from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
from playsound import playsound
import time
import dlib
import cv2
import os
import sys

# Function to play alarm sound
def sound_alarm():
    playsound('alarm.wav')  # Updated to match your filename

# Function to calculate Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
COUNTER = 0
ALARM_ON = False

# Detect working camera
def get_working_camera_index():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1

# Select a working webcam
cam_index = get_working_camera_index()
if cam_index == -1:
    print("[ERROR] No working webcam found.")
    sys.exit()

vs = cv2.VideoCapture(cam_index)
time.sleep(1.0)

# Load model and alarm sound
predictor_path = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(predictor_path):
    print(f"[ERROR] Model file not found: {predictor_path}")
    sys.exit()

if not os.path.exists('alarm.wav'):
    print("[ERROR] Alarm sound file 'alarm.wav' not found.")
    sys.exit()

# Initialize dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] Starting video stream...")

while True:
    ret, frame = vs.read()
    if not ret or frame is None:
        print("[ERROR] Failed to grab frame from camera.")
        break

    if frame.dtype != np.uint8:
        print(f"[ERROR] Frame is not 8-bit. Got: {frame.dtype}")
        break

    try:
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Validate 'gray' before using it
        if gray is None or not isinstance(gray, np.ndarray):
            print("[ERROR] gray is invalid or None.")
            break
        if gray.dtype != np.uint8:
            print(f"[ERROR] gray dtype is not uint8. Got: {gray.dtype}")
            break

    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        break

    # Safe to use now
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Drowsiness detection logic
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    Thread(target=sound_alarm, daemon=True).start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        # Display EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()


