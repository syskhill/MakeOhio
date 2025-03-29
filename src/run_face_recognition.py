#!/usr/bin/env python3
"""
Face Recognition Runner

This script loads a pre-trained face recognition model and runs face recognition
without training. It requires a model trained by train_face_model.py.
"""

import cv2
import numpy as np
import time
import serial
import os
import json
import sys
import logging
import argparse
import threading
import pickle
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_face_recognition")

# ================ Configuration ================
CAMERA_ID = 0  # Default camera (0 is usually the built-in webcam)
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
RECOGNITION_CONFIDENCE_THRESHOLD = 20  # Lower threshold to detect faces with low confidence
MODEL_SAVE_PATH = "face_recognition_model.yml"
MAPPING_SAVE_PATH = "patient_mapping.pkl"
ARDUINO_PORT = "/dev/ttyACM0"  # Default Arduino port
ARDUINO_BAUD_RATE = 9600

# ================ Initialize Face Detection ================
def initialize_face_detection():
    """Initialize the face detection cascade"""
    try:
        cascade_path = cv2.data.haarcascades + FACE_CASCADE_PATH
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error(f"Failed to load face cascade classifier")
            return None
        
        logger.info("Face detection initialized successfully")
        return face_cascade
    except Exception as e:
        logger.error(f"Error initializing face detection: {e}")
        return None

# ================ Connect to Arduino ================
def connect_to_arduino():
    """Try to connect to Arduino on various ports"""
    global ARDUINO_PORT
    
    possible_ports = [
        # Linux ports
        "/dev/ttyACM0", "/dev/ttyACM1", 
        "/dev/ttyUSB0", "/dev/ttyUSB1",
        # Windows ports
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8"
    ]
    
    # Add MacOS-specific ports
    if sys.platform == 'darwin':
        for i in range(10):
            possible_ports.append(f"/dev/tty.usbmodem{i+1}")
            possible_ports.append(f"/dev/tty.usbserial{i+1}")
    
    # Start with the configured port
    if ARDUINO_PORT:
        try:
            logger.info(f"Trying to connect to Arduino on configured port {ARDUINO_PORT}...")
            arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
            logger.info(f"Connected to Arduino on {ARDUINO_PORT}")
            time.sleep(2)  # Wait for Arduino to initialize
            return arduino
        except Exception as e:
            logger.warning(f"Could not connect to Arduino on {ARDUINO_PORT}: {e}")
    
    # Try all other ports
    logger.info(f"Searching for Arduino on ports: {possible_ports}")
    
    for port in possible_ports:
        if port == ARDUINO_PORT:
            continue  # Skip the one we already tried
            
        try:
            logger.info(f"Trying to connect to Arduino on {port}...")
            arduino = serial.Serial(port, ARDUINO_BAUD_RATE, timeout=1)
            # Update the port for future reference
            ARDUINO_PORT = port
            logger.info(f"Connected to Arduino on {port}")
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Test communication by sending a simple command
            try:
                arduino.write(b"MESSAGE:Face Recognition,System Started\n")
                logger.info("Sent test message to Arduino")
            except Exception as e:
                logger.warning(f"Test communication error: {e}")
            
            return arduino
        except Exception as e:
            logger.debug(f"Could not connect to {port}: {e}")
    
    logger.warning("Could not connect to Arduino on any known port")
    return None

def send_command_to_arduino(arduino, command):
    """Send a command to the Arduino"""
    if arduino:
        try:
            # First flush any existing data
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()
            
            # Send the command with newline terminator
            arduino.write(f"{command}\n".encode())
            
            # Wait for Arduino to process
            time.sleep(0.1)
            
            # Log the command
            logger.info(f"Sent command to Arduino: {command}")
            
            # Read any response (for debugging)
            try:
                response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
                if response.strip():
                    logger.info(f"Arduino response: {response.strip()}")
            except:
                pass
                
            return True
        except Exception as e:
            logger.error(f"Error sending command to Arduino: {e}")
            
            # Try to reconnect
            try:
                arduino.close()
            except:
                pass
                
            # Try to reconnect
            logger.info("Attempting to reconnect to Arduino...")
            try:
                arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
                logger.info(f"Reconnected to Arduino on {ARDUINO_PORT}")
                
                # Try sending the command again
                arduino.write(f"{command}\n".encode())
                logger.info(f"Re-sent command to Arduino after reconnection: {command}")
                return True
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect to Arduino: {reconnect_error}")
                return False
    else:
        logger.warning(f"Cannot send command, Arduino not connected: {command}")
        return False

# ================ Load Pre-trained Model ================
def load_trained_model_and_mapping():
    """Load the pre-trained model and patient mapping"""
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_SAVE_PATH):
            logger.error(f"Model file not found: {MODEL_SAVE_PATH}")
            return None, {}
        
        # Check if mapping file exists
        if not os.path.exists(MAPPING_SAVE_PATH):
            logger.error(f"Mapping file not found: {MAPPING_SAVE_PATH}")
            return None, {}
        
        # Load the model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_SAVE_PATH)
        logger.info(f"Loaded face recognition model from {MODEL_SAVE_PATH}")
        
        # Load the mapping
        with open(MAPPING_SAVE_PATH, 'rb') as f:
            label_to_patient_map = pickle.load(f)
        logger.info(f"Loaded patient mapping from {MAPPING_SAVE_PATH} with {len(label_to_patient_map)} patients")
        
        return recognizer, label_to_patient_map
    except Exception as e:
        logger.error(f"Error loading model or mapping: {e}")
        return None, {}

# ================ Load Patient Information ================
def load_patient_information():
    """Load basic patient information (no photos) for display purposes"""
    try:
        # Try loading from a JSON file if present
        patient_info_path = "patient_info.json"
        if os.path.exists(patient_info_path):
            with open(patient_info_path, 'r') as f:
                patients = json.load(f)
            logger.info(f"Loaded patient information from {patient_info_path} with {len(patients)} patients")
            return patients
        
        # If JSON file not present, try loading from pickle
        if os.path.exists(MAPPING_SAVE_PATH):
            with open(MAPPING_SAVE_PATH, 'rb') as f:
                label_to_patient_map = pickle.load(f)
            
            # Create simple patient records with just IDs
            patients = []
            for label, patient_id in label_to_patient_map.items():
                patients.append({
                    "id": patient_id,
                    "name": f"Patient {patient_id}",  # Default name
                    "pillTimes": "8:00,12:00,18:00",  # Default times
                    "slotNumber": 1  # Default slot
                })
            
            logger.info(f"Created basic patient information from mapping with {len(patients)} patients")
            return patients
        
        logger.warning("No patient information found. Using empty list.")
        return []
    except Exception as e:
        logger.error(f"Error loading patient information: {e}")
        return []

def check_patient_access(patient_id, patients):
    """Check if the patient has access at the current time"""
    # Find the patient in the list
    patient = None
    for p in patients:
        if p["id"] == patient_id:
            patient = p
            break
    
    if not patient:
        logger.warning(f"Patient not found: {patient_id}")
        return False, "Patient not found", 0
    
    # Get current time
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    
    # Parse pill times (format expected: "8:00,12:00,18:00")
    pill_time_strings = patient.get("pillTimes", "").split(",")
    pill_time_strings = [t.strip() for t in pill_time_strings if t.strip()]
    
    if not pill_time_strings:
        # If no pill times are specified, allow access (for testing)
        logger.info(f"No pill times specified for patient {patient.get('name')}, allowing access")
        return True, "Access granted (no pill times specified)", patient.get("slotNumber", 0)
    
    # Check if current time is within ±60 minutes of any pill time
    for time_str in pill_time_strings:
        try:
            hour, minute = map(int, time_str.split(":"))
            
            # Calculate difference in minutes
            diff_minutes = abs(
                (current_hour * 60 + current_minute) - (hour * 60 + minute)
            )
            
            # Allow access if within ±60 minutes
            if diff_minutes <= 60:
                logger.info(f"Access granted for patient {patient.get('name')} (within {diff_minutes} minutes of {time_str})")
                return True, f"Access granted (near pill time {time_str})", patient.get("slotNumber", 0)
        except Exception as e:
            logger.error(f"Error parsing pill time '{time_str}': {e}")
    
    # No matching pill time found
    logger.info(f"Access denied for patient {patient.get('name')} (not within pill time window)")
    return False, "Access denied: Not within pill time window", patient.get("slotNumber", 0)

# ================ Face Recognition Loop ================
def run_face_recognition(face_cascade, recognizer, label_to_patient_map, patients, arduino):
    """Run continuous face recognition"""
    # Check if we have a trained model
    if not face_cascade or not recognizer or not label_to_patient_map:
        logger.error("Missing required components for face recognition")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        logger.error(f"Error: Could not open camera {CAMERA_ID}")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Variables for recognition
    last_recognition_time = 0
    recognition_cooldown = 1  # Seconds between recognitions
    
    logger.info("Starting face recognition loop")
    if arduino:
        send_command_to_arduino(arduino, "MESSAGE:Face Recognition,Running...")
    
    running = True
    frame_count = 0
    fps_time = time.time()
    
    try:
        while running:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                logger.error("Error reading from camera")
                time.sleep(1)
                continue
            
            # FPS calculation
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                logger.debug(f"FPS: {fps:.1f}")
                frame_count = 0
                fps_time = current_time
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=4,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Check if enough time has passed since last recognition
                if current_time - last_recognition_time > recognition_cooldown:
                    # Try to recognize the face
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Apply histogram equalization for better contrast
                    face_roi = cv2.equalizeHist(face_roi)
                    
                    # Resize to consistent size
                    resized_face = cv2.resize(face_roi, (100, 100))
                    
                    try:
                        # Get prediction
                        label, confidence = recognizer.predict(resized_face)
                        
                        # Convert confidence to a more intuitive 0-100 scale (100 is best match)
                        confidence_score = max(0, 100 - min(100, confidence))
                        
                        # Display confidence score on frame
                        cv2.putText(
                            frame,
                            f"Conf: {confidence_score:.1f}%",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )
                        
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
                                
                                logger.info(f"Recognized: {patient_name} (ID: {patient_id}, Confidence: {confidence_score:.1f}%)")
                                
                                # Check if patient has access
                                has_access, message, slot_number = check_patient_access(patient_id, patients)
                                
                                # Display recognition result on frame
                                status_color = (0, 255, 0) if has_access else (0, 0, 255)
                                status_text = "Access Granted" if has_access else "Access Denied"
                                
                                cv2.putText(
                                    frame,
                                    f"{patient_name}",
                                    (x, y+h+20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    status_color,
                                    2
                                )
                                cv2.putText(
                                    frame,
                                    status_text,
                                    (x, y+h+45),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    status_color,
                                    2
                                )
                                
                                # Send command to Arduino with confidence score included
                                if arduino:
                                    if has_access:
                                        logger.info(f"Access granted for slot {slot_number} with confidence {confidence_score:.1f}%")
                                        send_command_to_arduino(arduino, f"ACCESS:{patient_id},{patient_name},{slot_number},{confidence_score:.1f}")
                                    else:
                                        logger.info(f"Access denied: {message} with confidence {confidence_score:.1f}%")
                                        send_command_to_arduino(arduino, f"DENY:{patient_id},{message},{confidence_score:.1f}")
                                
                                last_recognition_time = current_time
                    except Exception as e:
                        logger.error(f"Error during face recognition: {e}")
            
            # Add debugging overlays
            debug_mode = os.environ.get('PILL_DISPENSER_DEBUG', '0') == '1'
            if debug_mode:
                # Show FPS
                cv2.putText(
                    frame,
                    f"FPS: {frame_count/max(0.1, elapsed):.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                # Show found faces count
                cv2.putText(
                    frame,
                    f"Faces: {len(faces)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                # Show patient count
                cv2.putText(
                    frame,
                    f"Patients: {len(patients)}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            # Display the frame
            window_title = "Face Recognition - DEBUG MODE" if debug_mode else "Face Recognition"
            cv2.imshow(window_title, frame)
            
            # Check for key press (q to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                running = False
            
            # Send heartbeat message to Arduino every 30 seconds
            if int(current_time) % 30 == 0 and abs(current_time - int(current_time)) < 0.1:
                if arduino:
                    heartbeat_msg = f"Heartbeat,{time.strftime('%H:%M:%S')}"
                    send_command_to_arduino(arduino, f"MESSAGE:{heartbeat_msg}")
    
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Error in face recognition loop: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        if arduino:
            send_command_to_arduino(arduino, "MESSAGE:System Stopped,Goodbye!")
            time.sleep(1)
            arduino.close()
        
        logger.info("Face recognition stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Face Recognition Runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID to use")
    parser.add_argument("--threshold", type=float, default=20, help="Recognition confidence threshold")
    parser.add_argument("--model", default="face_recognition_model.yml", help="Path to trained model")
    parser.add_argument("--mapping", default="patient_mapping.pkl", help="Path to patient mapping")
    parser.add_argument("--arduino-port", help="Arduino serial port")
    parser.add_argument("--no-arduino", action="store_true", help="Run without Arduino connection")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Update configuration from arguments
    global CAMERA_ID, RECOGNITION_CONFIDENCE_THRESHOLD, MODEL_SAVE_PATH, MAPPING_SAVE_PATH, ARDUINO_PORT
    
    CAMERA_ID = args.camera
    RECOGNITION_CONFIDENCE_THRESHOLD = args.threshold
    MODEL_SAVE_PATH = args.model
    MAPPING_SAVE_PATH = args.mapping
    
    if args.arduino_port:
        ARDUINO_PORT = args.arduino_port
    
    # Set debug mode environment variable
    if args.debug:
        os.environ['PILL_DISPENSER_DEBUG'] = '1'
        logger.info("Debug mode enabled")
    
    logger.info("Face Recognition Runner")
    logger.info("======================")
    logger.info(f"Camera ID: {CAMERA_ID}")
    logger.info(f"Recognition threshold: {RECOGNITION_CONFIDENCE_THRESHOLD}")
    logger.info(f"Model path: {MODEL_SAVE_PATH}")
    logger.info(f"Mapping path: {MAPPING_SAVE_PATH}")
    
    # Initialize face detection
    face_cascade = initialize_face_detection()
    if not face_cascade:
        logger.error("Failed to initialize face detection. Exiting.")
        return
    
    # Load pre-trained model and mapping
    recognizer, label_to_patient_map = load_trained_model_and_mapping()
    if not recognizer or not label_to_patient_map:
        logger.error("Failed to load pre-trained model and mapping. Exiting.")
        logger.error("Please run train_face_model.py first to create the model.")
        return
    
    # Load patient information
    patients = load_patient_information()
    
    # Connect to Arduino (unless disabled)
    arduino = None
    if not args.no_arduino:
        arduino = connect_to_arduino()
        if not arduino:
            logger.warning("Failed to connect to Arduino. Running without Arduino support.")
    else:
        logger.info("Arduino connection disabled by command line argument")
    
    # Run face recognition
    run_face_recognition(face_cascade, recognizer, label_to_patient_map, patients, arduino)

if __name__ == "__main__":
    main()