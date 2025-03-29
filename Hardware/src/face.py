import cv2
import face_recognition
import numpy as np
import serial
import time
from datetime import datetime

arduino = serial.Serial('COM4',9600,timeout=1) 

known_face_encodings = []
known_face_names = ["Sam"]

user_image = face_recognition.load_image_file('src\\samC.jpg')
user_encoding = face_recognition.face_encodings(user_image)[0]
known_face_encodings.append(user_encoding)

video_capture = cv2.VideoCapture(0)

def is_correct_time():
    return True

def recognize_face():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        return False
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        if True in matches:
            print("Face recognized!")
            return True

    print("Face not recognized")
    return False
while True:
    if recognize_face() and is_correct_time():
        print("Access granted! Unlocking dispenser...")
        arduino.write(b'UNLOCK\n')  # Send unlock command to Arduino
        time.sleep(10)  # Avoid repeated unlocks within a short time
    else:
        print("Access denied")

    time.sleep(3)  # Small delay before next recognition attempt

video_capture.release()
cv2.destroyAllWindows()