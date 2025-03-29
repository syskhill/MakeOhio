import cv2
import face_recognition
import numpy as np
import serial
import time
import os
from datetime import datetime

# Try different serial ports based on platform
try:
    # Try Linux/Mac ports first
    port = '/dev/ttyACM0'  # Common Arduino port on Linux
    arduino = serial.Serial(port, 9600, timeout=1)
    print(f"Connected to {port}")
except:
    try:
        port = '/dev/ttyUSB0'  # Another common Arduino port on Linux
        arduino = serial.Serial(port, 9600, timeout=1)
        print(f"Connected to {port}")
    except:
        try:
            # Fall back to Windows
            port = 'COM4'
            arduino = serial.Serial(port, 9600, timeout=1)
            print(f"Connected to {port}")
        except:
            print("Could not connect to Arduino! Check connection and port.")
            arduino = None

known_face_encodings = []
known_face_names = ["Sam"]

# Fix path for cross-platform compatibility
image_path = os.path.join('src', 'samC.jpg')
try:
    user_image = face_recognition.load_image_file(image_path)
    user_encoding = face_recognition.face_encodings(user_image)[0]
    known_face_encodings.append(user_encoding)
    print(f"Successfully loaded face image from {image_path}")
except Exception as e:
    print(f"Error loading face image: {e}")
    exit(1)

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
        if arduino:
            arduino.write(b'UNLOCK\n')
            print("Sent UNLOCK command to Arduino")
        else:
            print("Arduino not connected - cannot send UNLOCK command")
        time.sleep(10)  # Cooldown period
    elif result is False:
        print("Access denied! Locking dispenser.")
        if arduino:
            arduino.write(b'LOCK\n')
            print("Sent LOCK command to Arduino")
        else:
            print("Arduino not connected - cannot send LOCK command")
    else:
        print("No face found. Standing by...")

    # Display the current frame with face recognition boxes
    # This is useful for debugging
    ret, frame = video_capture.read()
    if ret:
        # Display face boxes
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    time.sleep(0.1)  # Small delay to prevent CPU overuse

# Clean up resources
if arduino:
    arduino.close()
video_capture.release()
cv2.destroyAllWindows()
print("Program ended")
