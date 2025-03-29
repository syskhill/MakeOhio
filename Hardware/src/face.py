import cv2
import face_recognition
import numpy as np
import serial
import time
from datetime import datetime

arduino = serial.Serial('COM4', 9600, timeout=1)

known_face_encodings = []
known_face_names = ["Sam"]

user_image = face_recognition.load_image_file('src\\samC.jpg')
user_encoding = face_recognition.face_encodings(user_image)[0]
known_face_encodings.append(user_encoding)

video_capture = cv2.VideoCapture(0)

def is_correct_time():
    return True  # Customize your time check logic here

def recognize_face():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        return None  # Nothing to do

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        print("No face detected")
        return None

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        if True in matches:
            print("Face recognized!")
            return True

    print("Unrecognized face detected")
    return False  # Face was detected but not matched

while True:
    result = recognize_face()

    if result is True and is_correct_time():
        print("Access granted! Unlocking dispenser...")
        arduino.write(b'UNLOCK\n')
        time.sleep(10)  # Cooldown period
    elif result is False:
        print("Access denied! Locking dispenser.")
        arduino.write(b'LOCK\n')
    else:
        print("No face found. Standing by...")

    time.sleep(3)

video_capture.release()
cv2.destroyAllWindows()
