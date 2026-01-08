import cv2
import os
import numpy as np
import pickle

DATASET_DIR = "dataset"
MODEL_DIR = "encodings"
MODEL_PATH = os.path.join(MODEL_DIR, "lbph_model.yml")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

label_id = 0

print("[INFO] Training LBPH model...")

for student_folder in os.listdir(DATASET_DIR):
    student_path = os.path.join(DATASET_DIR, student_folder)

    if not os.path.isdir(student_path):
        continue

    label_map[label_id] = student_folder

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        detected = face_detector.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in detected:
            face = img[y:y+h, x:x+w]
            faces.append(face)
            labels.append(label_id)

    label_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_PATH)

with open(LABEL_MAP_PATH, "wb") as f:
    pickle.dump(label_map, f)

print("✅ LBPH model trained successfully.")
