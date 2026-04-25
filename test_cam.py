import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not access webcam.")
else:
    print("[INFO] Webcam opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Frame capture failed.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
