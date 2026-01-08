import cv2
import pickle
import os
from datetime import datetime

MODEL_PATH = "encodings/lbph_model.yml"
LABEL_MAP_PATH = "encodings/label_map.pkl"

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

cam = cv2.VideoCapture(0)

print("[INFO] Starting face recognition...")

recognized_today = set()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        name = "Unknown"

        if confidence < 70:  # lower = better in LBPH
            name = label_map.get(label, "Unknown")

            if name not in recognized_today:
                recognized_today.add(name)
                print(f"✅ Attendance marked for {name} at {datetime.now()}")

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{name} ({round(confidence,2)})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Smart Attendance - LBPH", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
