import cv2
import os

def capture_faces(student_id, student_name, save_path="dataset"):
    """
    Captures face images for a student, one at a time,
    when the user presses 's'.
    """
    cam = cv2.VideoCapture(0)
    student_folder = os.path.join(save_path, f"{student_id}_{student_name}")
    os.makedirs(student_folder, exist_ok=True)

    count = 0
    print("🖼️ Press 's' to take a photo, 'q' to quit registration.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Camera error")
            break

        cv2.imshow("Register - Press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            img_path = os.path.join(student_folder, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"📸 Saved {img_path}")
            count += 1

    cam.release()
    cv2.destroyAllWindows()
    print(f"✅ Registration complete. Total images saved: {count}")

if __name__ == "__main__":
    student_id = input("Enter Student ID: ").strip()
    student_name = input("Enter Student Name: ").strip()
    capture_faces(student_id, student_name)
