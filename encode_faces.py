import face_recognition
import cv2
import os
import pickle

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings/face_encodings.pkl"

os.makedirs("encodings", exist_ok=True)

known_encodings = []
known_names = []

print("[INFO] Encoding faces...")

for student_folder in os.listdir(DATASET_DIR):
    student_path = os.path.join(DATASET_DIR, student_folder)

    if not os.path.isdir(student_path):
        continue

    for image_name in os.listdir(student_path):
        image_path = os.path.join(student_path, image_name)

        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(student_folder)

print(f"[INFO] Total faces encoded: {len(known_encodings)}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("✅ Face encodings saved successfully.")
