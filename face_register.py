import cv2
import os

def capture_faces(student_id, student_name, save_path="dataset", max_images=20):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    student_folder = os.path.join(save_path, f"{student_id}_{student_name}")

    if not os.path.exists(student_folder):
        os.makedirs(student_folder)

    print("[INFO] Starting face capture. Look at the camera.")

    while count < max_images:
        ret, frame = cam.read()
        if not ret:
            print("❌ Camera error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            img_path = os.path.join(student_folder, f"img_{count+1}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1
            print(f"📸 Captured {count}/{max_images}")

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("✅ Face registration completed.")

if __name__ == "__main__":
    sid = input("Enter Student ID: ")
    name = input("Enter Name: ")
    capture_faces(sid, name)
