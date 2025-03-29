import cv2
import numpy as np
import time
import serial
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import threading
import logging

# ================ Configuration ================
API_PORT = 5050
ARDUINO_PORT = "/dev/ttyACM0"  # Common Arduino port on Linux, change as needed
ARDUINO_BAUD_RATE = 9600
CAMERA_ID = 0  # Default camera (0 is usually the built-in webcam)
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
RECOGNITION_CONFIDENCE_THRESHOLD = 70  # Minimum confidence (0-100)
PHOTOS_DIR = "photos"  # Directory to store patient photos
MODEL_SAVE_PATH = "face_recognition_model.yml"  # Path to save/load trained model

# Create photos directory if it doesn't exist
os.makedirs(PHOTOS_DIR, exist_ok=True)

# ================ Setup Logging ================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_recognition")

# ================ Initialize Face Detection ================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ================ Global Variables ================
patients = []  # List to store patient data
patient_faces = {}  # Dictionary to store face encodings by patient ID
label_to_patient_map = {}  # Map numeric labels to patient IDs
recognition_active = False  # Flag to control face recognition thread
face_recognition_thread = None  # Thread for face recognition
last_recognition_result = None  # Store the last recognition result

# ================ Connect to Arduino ================
arduino = None
try:
    # Try different serial ports based on platform
    possible_ports = [
        "/dev/ttyACM0",  # Common Arduino port on Linux
        "/dev/ttyUSB0",  # Another common Arduino port on Linux
        "COM4"           # Common port on Windows
    ]
    
    for port in possible_ports:
        try:
            arduino = serial.Serial(port, ARDUINO_BAUD_RATE, timeout=1)
            logger.info(f"Connected to Arduino on {port}")
            time.sleep(2)  # Wait for Arduino to initialize
            break
        except:
            continue
            
    if arduino is None:
        logger.warning("Could not connect to Arduino on any known port")
except Exception as e:
    logger.error(f"Error initializing Arduino connection: {e}")

# ================ Initialize Flask API ================
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"status": "running", "message": "Face Recognition API"})

@app.route('/photos/<path:filename>')
def get_photo(filename):
    return send_from_directory(PHOTOS_DIR, filename)

@app.route('/record/arduino/patients', methods=['GET'])
def get_patients():
    return jsonify(patients)

@app.route('/record/arduino/patients', methods=['POST'])
def add_patient():
    data = request.json
    # Validate required fields
    if not all(k in data for k in ['id', 'name', 'slotNumber']):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Add patient to the list
    patients.append(data)
    
    # Save patient data to file
    save_patient_data()
    
    return jsonify({"status": "success", "message": "Patient added"}), 201

@app.route('/record/arduino/patients/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    global patients
    initial_count = len(patients)
    patients = [p for p in patients if p['id'] != patient_id]
    
    if len(patients) < initial_count:
        save_patient_data()
        return jsonify({"status": "success", "message": "Patient deleted"})
    else:
        return jsonify({"error": "Patient not found"}), 404

@app.route('/record/arduino/access/<patient_id>', methods=['GET'])
def check_access(patient_id):
    # Find the patient
    patient = next((p for p in patients if p['id'] == patient_id), None)
    
    if not patient:
        return jsonify({"access": False, "message": "Patient not found", "slotNumber": 0})
    
    # In a real system, you would check if the current time is within a valid window
    # For this example, we'll always grant access
    return jsonify({
        "access": True,
        "message": "Access granted",
        "slotNumber": patient.get('slotNumber', 1)
    })

@app.route('/record/arduino/upload-photo/<patient_id>', methods=['POST'])
def upload_photo(patient_id):
    if 'photo' not in request.files:
        return jsonify({"error": "No photo provided"}), 400
    
    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No photo selected"}), 400
    
    # Save the photo
    filename = f"{patient_id}.jpg"
    file_path = os.path.join(PHOTOS_DIR, filename)
    file.save(file_path)
    
    # Update patient record with photo path
    for patient in patients:
        if patient['id'] == patient_id:
            patient['photo'] = filename
            break
    
    save_patient_data()
    
    # Retrain the face recognition model
    load_patient_photos()
    train_recognizer()
    
    return jsonify({"status": "success", "message": "Photo uploaded"})

@app.route('/record/arduino/status', methods=['GET'])
def get_status():
    status = {
        "arduino_connected": arduino is not None,
        "recognition_active": recognition_active,
        "patients_loaded": len(patients),
        "last_recognition": last_recognition_result
    }
    return jsonify(status)

@app.route('/record/arduino/start-recognition', methods=['POST'])
def start_recognition():
    global recognition_active, face_recognition_thread
    
    if recognition_active:
        return jsonify({"status": "already_running", "message": "Face recognition is already running"})
    
    recognition_active = True
    face_recognition_thread = threading.Thread(target=run_face_recognition)
    face_recognition_thread.daemon = True
    face_recognition_thread.start()
    
    return jsonify({"status": "started", "message": "Face recognition started"})

@app.route('/record/arduino/stop-recognition', methods=['POST'])
def stop_recognition():
    global recognition_active
    
    if not recognition_active:
        return jsonify({"status": "not_running", "message": "Face recognition is not running"})
    
    recognition_active = False
    # The thread will exit on its own when it checks the flag
    
    return jsonify({"status": "stopped", "message": "Face recognition stopping"})

# ================ Helper Functions ================
def save_patient_data():
    """Save patient data to a JSON file"""
    with open("patients.json", 'w') as f:
        json.dump(patients, f, indent=2)
    logger.info(f"Saved {len(patients)} patients to database")

def load_patient_data():
    """Load patient data from JSON file"""
    global patients
    try:
        if os.path.exists("patients.json"):
            with open("patients.json", 'r') as f:
                patients = json.load(f)
            logger.info(f"Loaded {len(patients)} patients from database")
        else:
            logger.warning("No patient database found. Starting with empty database.")
            patients = []
    except Exception as e:
        logger.error(f"Error loading patient data: {e}")
        patients = []

def load_patient_photos():
    """Load and process patient photos for face recognition"""
    global patient_faces
    patient_faces = {}
    
    for patient in patients:
        patient_id = patient['id']
        photo_filename = patient.get('photo')
        
        if not photo_filename:
            logger.warning(f"No photo specified for patient {patient['name']} ({patient_id})")
            continue
        
        # Check if photo exists
        file_path = os.path.join(PHOTOS_DIR, photo_filename)
        if not os.path.exists(file_path):
            logger.warning(f"Photo file not found for patient {patient['name']} ({patient_id}): {file_path}")
            continue
        
        # Load and process the photo
        try:
            img = cv2.imread(file_path)
            if img is None:
                logger.warning(f"Failed to read image file for patient {patient['name']} ({patient_id})")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Take the first face found
                face_roi = gray[y:y+h, x:x+w]
                
                # Add to training data
                if patient_id not in patient_faces:
                    patient_faces[patient_id] = []
                
                patient_faces[patient_id].append(face_roi)
                logger.info(f"Loaded face for {patient['name']} ({patient_id})")
            else:
                logger.warning(f"No face detected in photo for patient {patient['name']} ({patient_id})")
        except Exception as e:
            logger.error(f"Error processing photo for patient {patient['name']} ({patient_id}): {e}")

def train_recognizer():
    """Train the face recognizer with loaded patient faces"""
    global label_to_patient_map
    
    if not patient_faces:
        logger.warning("No face data available to train the recognizer")
        return False
    
    faces = []
    labels = []
    label_to_patient_map = {}  # Map numeric labels to patient IDs
    
    next_label = 0
    for patient_id, face_list in patient_faces.items():
        label_to_patient_map[next_label] = patient_id
        
        for face in face_list:
            # Resize face if needed
            resized_face = cv2.resize(face, (100, 100))
            faces.append(resized_face)
            labels.append(next_label)
        
        next_label += 1
    
    if not faces:
        logger.warning("No valid faces to train the recognizer")
        return False
        
    recognizer.train(faces, np.array(labels))
    logger.info(f"Trained recognizer with {len(faces)} faces from {len(patient_faces)} patients")
    
    # Save the trained model
    try:
        recognizer.save(MODEL_SAVE_PATH)
        logger.info(f"Saved face recognition model to {MODEL_SAVE_PATH}")
    except Exception as e:
        logger.error(f"Error saving face recognition model: {e}")
    
    return True

def load_trained_model():
    """Load a previously trained model if available"""
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            recognizer.read(MODEL_SAVE_PATH)
            logger.info(f"Loaded face recognition model from {MODEL_SAVE_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error loading face recognition model: {e}")
    return False

def send_command_to_arduino(command):
    """Send a command to the Arduino"""
    global arduino
    
    if arduino:
        try:
            arduino.write(f"{command}\n".encode())
            time.sleep(0.1)  # Give Arduino time to process
            logger.info(f"Sent command to Arduino: {command}")
            return True
        except Exception as e:
            logger.error(f"Error sending command to Arduino: {e}")
            
            # Try to reconnect
            try:
                arduino.close()
            except:
                pass
                
            try:
                arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
                logger.info(f"Reconnected to Arduino on {ARDUINO_PORT}")
                arduino.write(f"{command}\n".encode())
                return True
            except Exception as e:
                logger.error(f"Failed to reconnect to Arduino: {e}")
                arduino = None
    else:
        logger.warning(f"Cannot send command, Arduino not connected: {command}")
    
    return False

def run_face_recognition():
    """Run continuous face recognition in a separate thread"""
    global recognition_active, last_recognition_result
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        recognition_active = False
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_recognition_time = 0
    recognition_cooldown = 5  # Seconds between recognitions
    
    logger.info("Starting face recognition thread")
    
    while recognition_active:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error reading from camera")
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
            # Draw rectangle for visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Check if enough time has passed since last recognition
            if current_time - last_recognition_time > recognition_cooldown:
                # Try to recognize the face
                face_roi = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (100, 100))
                
                try:
                    label, confidence = recognizer.predict(resized_face)
                    
                    # Convert confidence to a more intuitive 0-100 scale (100 is best match)
                    confidence_score = 100 - min(100, confidence)
                    
                    # Display confidence score on frame
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
                            
                            logger.info(f"Recognized: {patient_name} (ID: {patient_id}, Confidence: {confidence_score:.1f}%)")
                            
                            # Store recognition result
                            last_recognition_result = {
                                "patient_id": patient_id,
                                "patient_name": patient_name,
                                "confidence": confidence_score,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # Check if patient has access
                            slot_number = 1  # Default
                            for p in patients:
                                if p['id'] == patient_id:
                                    slot_number = p.get('slotNumber', 1)
                                    break
                            
                            # In a real application, check access times
                            has_access = True  # Simplified for demo
                            
                            # Display recognition result on frame
                            status_color = (0, 255, 0) if has_access else (0, 0, 255)
                            status_text = "Access Granted" if has_access else "Access Denied"
                            
                            cv2.putText(frame, f"{patient_name}", 
                                        (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                            cv2.putText(frame, status_text, 
                                        (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                            
                            # Send command to Arduino
                            if has_access:
                                logger.info(f"Access granted for slot {slot_number}")
                                send_command_to_arduino(f"ACCESS:{patient_id},{slot_number}")
                            else:
                                logger.info("Access denied")
                                send_command_to_arduino(f"DENY:{patient_id}")
                                
                            last_recognition_time = current_time
                except Exception as e:
                    logger.error(f"Error during face recognition: {e}")
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        # Break loop on 'q' press or if recognition_active is False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recognition_active = False
            break
        
        # Small delay to prevent high CPU usage
        time.sleep(0.05)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Face recognition thread stopped")

# ================ Main Function ================
def main():
    """Main function"""
    logger.info("Pill Dispenser Face Recognition System")
    logger.info("--------------------------------------")
    
    # Load patient data
    load_patient_data()
    
    # Load trained model or train a new one
    model_loaded = load_trained_model()
    
    if not model_loaded or len(patients) > 0:
        # Load patient photos and train the recognizer
        load_patient_photos()
        train_recognizer()
    
    # Start the web server in a separate thread
    server_thread = threading.Thread(target=app.run, kwargs={
        'host': '0.0.0.0',
        'port': API_PORT,
        'debug': False
    })
    server_thread.daemon = True
    server_thread.start()
    
    logger.info(f"API server running at http://localhost:{API_PORT}")
    
    # Start face recognition automatically
    global recognition_active, face_recognition_thread
    recognition_active = True
    face_recognition_thread = threading.Thread(target=run_face_recognition)
    face_recognition_thread.daemon = True
    face_recognition_thread.start()
    
    # Keep main thread alive to handle keyboard interrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        recognition_active = False
        
        # Clean up Arduino connection
        if arduino:
            arduino.close()
        
        logger.info("Application terminated")

if __name__ == "__main__":
    main()