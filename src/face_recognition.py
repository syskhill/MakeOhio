import cv2
import numpy as np
import requests
import time
import serial
import os
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:5050/record/arduino"
ARDUINO_PORT = "/dev/ttyUSB0"  # Change this to your Arduino serial port
ARDUINO_BAUD_RATE = 9600
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
RECOGNITION_CONFIDENCE_THRESHOLD = 70  # Minimum confidence to consider a face recognized (0-100)
CAMERA_ID = 0  # Default camera (0 is usually the built-in webcam)

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Variables for patient data
patients = []
patient_faces = {}  # Dictionary to store face encodings by patient ID

# Connect to Arduino (if available)
arduino = None
try:
    arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2)  # Wait for Arduino to initialize
except Exception as e:
    print(f"Warning: Could not connect to Arduino: {e}")
    print("Running in standalone mode (no Arduino control)")

def fetch_patient_data():
    """Fetch patient data from the API"""
    try:
        response = requests.get(f"{API_URL}/patients")
        if response.status_code == 200:
            global patients
            patients = response.json()
            print(f"Loaded {len(patients)} patients from database")
            return True
        else:
            print(f"Error fetching patient data: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False

def load_patient_photos():
    """Load and process patient photos for face recognition"""
    for patient in patients:
        patient_id = patient['id']
        photo_filename = patient['photo']
        
        # Check if we already have a valid photo path
        if not os.path.exists(photo_filename):
            # Look in common locations
            possible_paths = [
                photo_filename,
                f"photos/{photo_filename}",
                f"/home/connor/PlatformIO/Projects/Hardware/photos/{photo_filename}"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    photo_filename = path
                    break
            else:
                print(f"Warning: Could not find photo for patient {patient['name']} ({patient_id})")
                continue
        
        # Load and process the photo
        try:
            img = cv2.imread(photo_filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Take the first face found
                face_roi = gray[y:y+h, x:x+w]
                
                # Add to training data
                if patient_id not in patient_faces:
                    patient_faces[patient_id] = []
                
                patient_faces[patient_id].append(face_roi)
                print(f"Loaded face for {patient['name']} ({patient_id})")
            else:
                print(f"No face detected in photo for patient {patient['name']} ({patient_id})")
        except Exception as e:
            print(f"Error processing photo for patient {patient['name']} ({patient_id}): {e}")

def train_recognizer():
    """Train the face recognizer with loaded patient faces"""
    if not patient_faces:
        print("No face data available to train the recognizer")
        return False
    
    faces = []
    labels = []
    label_map = {}  # Map patient IDs to numeric labels
    
    next_label = 0
    for patient_id, face_list in patient_faces.items():
        label_map[next_label] = patient_id
        
        for face in face_list:
            faces.append(face)
            labels.append(next_label)
        
        next_label += 1
    
    recognizer.train(faces, np.array(labels))
    print(f"Trained recognizer with {len(faces)} faces from {len(patient_faces)} patients")
    
    # Save label map for recognition
    global label_to_patient_map
    label_to_patient_map = label_map
    return True

def check_patient_access(patient_id):
    """Check if the patient has access at the current time"""
    try:
        response = requests.get(f"{API_URL}/access/{patient_id}")
        if response.status_code == 200:
            data = response.json()
            return data.get("access", False), data.get("message", ""), data.get("slotNumber", 0)
        else:
            print(f"Error checking patient access: HTTP {response.status_code}")
            return False, "Error checking access", 0
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False, f"Connection error: {str(e)}", 0

def send_command_to_arduino(command):
    """Send a command to the Arduino"""
    if arduino:
        try:
            arduino.write(f"{command}\n".encode())
            time.sleep(0.1)  # Give Arduino time to process
            return True
        except Exception as e:
            print(f"Error sending command to Arduino: {e}")
            return False
    return False

def run_face_recognition():
    """Run continuous face recognition"""
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_recognition_time = 0
    recognition_cooldown = 5  # Seconds between recognitions
    
    print("Starting face recognition - press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        current_time = time.time()
        
        # Process each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Check if enough time has passed since last recognition
            if current_time - last_recognition_time > recognition_cooldown:
                # Try to recognize the face
                face_roi = gray[y:y+h, x:x+w]
                
                try:
                    label, confidence = recognizer.predict(face_roi)
                    
                    # Convert confidence to a more intuitive 0-100 scale (100 is best match)
                    confidence_score = 100 - min(100, confidence)
                    
                    # Display confidence score
                    cv2.putText(frame, f"Conf: {confidence_score:.1f}%", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Check if confidence meets threshold
                    if confidence_score >= RECOGNITION_CONFIDENCE_THRESHOLD:
                        if label in label_to_patient_map:
                            patient_id = label_to_patient_map[label]
                            
                            # Find patient name
                            patient_name = "Unknown"
                            for p in patients:
                                if p['id'] == patient_id:
                                    patient_name = p['name']
                                    break
                            
                            print(f"Recognized: {patient_name} (ID: {patient_id}, Confidence: {confidence_score:.1f}%)")
                            
                            # Check if patient has access at current time
                            has_access, message, slot_number = check_patient_access(patient_id)
                            
                            # Display recognition result
                            status_color = (0, 255, 0) if has_access else (0, 0, 255)
                            status_text = "Access Granted" if has_access else "Access Denied"
                            
                            cv2.putText(frame, f"{patient_name}", 
                                        (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                            cv2.putText(frame, status_text, 
                                        (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                            
                            # If access granted, send command to Arduino to dispense pills
                            if has_access:
                                print(f"Access granted for slot {slot_number}")
                                send_command_to_arduino(f"ACCESS:{patient_id},{slot_number}")
                            else:
                                print(f"Access denied: {message}")
                                send_command_to_arduino(f"DENY:{patient_id}")
                                
                            last_recognition_time = current_time
                except Exception as e:
                    print(f"Error during face recognition: {e}")
        
        # Display the resulting frame
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Face Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    print("Pill Dispenser Face Recognition System")
    print("--------------------------------------")
    
    # Fetch patient data
    if not fetch_patient_data():
        print("Failed to fetch patient data. Check your API connection.")
        return
    
    # Load patient photos
    load_patient_photos()
    
    # Train the face recognizer
    if not train_recognizer():
        print("Failed to train face recognizer. Check patient photos.")
        return
    
    # Run face recognition
    run_face_recognition()
    
    # Clean up
    if arduino:
        arduino.close()

if __name__ == "__main__":
    main()