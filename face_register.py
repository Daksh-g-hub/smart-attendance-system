import cv2
import os

def capture_faces(student_id, student_name, save_path="dataset"):
    cam = cv2.VideoCapture(0)
    count = 0
    student_folder = os.path.join(save_path, f"{student_id}_{student_name}")

    if not os.path.exists(student_folder):
        os.makedirs(student_folder)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("👎 Camera error")
            break

        cv2.imshow("Register - Press q to quit", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        # Save every 10th frame for dataset
        if count % 10 == 0:
            img_path = os.path.join(student_folder, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"📸 Saved {img_path}")

        count += 1

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sid = input("Enter Student ID: ")
    name = input("Enter Name: ")
    capture_faces(sid, name)
