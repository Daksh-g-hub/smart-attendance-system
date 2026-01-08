import os
import subprocess

def run_command(cmd):
    """Run a terminal command."""
    subprocess.call(cmd, shell=True)

def main():
    print("=== Smart Attendance System ===\n")

    # Step 1: Register new students
    choice = input("Do you want to register a new student? (y/n): ").lower()
    if choice == "y":
        print("Opening face registration...")
        run_command("python attendence_register.py")
    else:
        print("Skipping registration...\n")

    # Step 2: Encode faces / train model
    print("Training LBPH model...")
    run_command("python encode_faces.py")

    # Step 3: Start face recognition
    print("\nStarting real-time face recognition...")
    run_command("python recognize_faces.py")

if __name__ == "__main__":
    main()
