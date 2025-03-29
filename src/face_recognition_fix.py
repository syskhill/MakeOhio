#!/usr/bin/env python3
# Modified version of face_recognition.py that checks for required dependencies

import cv2
import numpy as np
import time
import serial
import os
import json
import requests
import sys
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import threading
import logging
import base64
from pymongo import MongoClient
from bson.objectid import ObjectId

# ================ Check Dependencies ================
def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    # Check OpenCV face module
    try:
        # Try to import face module
        face_module = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        missing_deps.append("opencv-contrib-python")
    
    # Check other dependencies if needed
    # ...
    
    if missing_deps:
        print("\nERROR: Missing required dependencies!\n")
        print("Please install the following packages:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        print("\nThen try running this script again.")
        return False
    
    return True

# Run dependency check
try:
    import cv2.face
    print("All dependencies are already installed!")
except ImportError:
    print("\nERROR: Missing required dependencies!\n")
    print("Please install the following package:")
    print("  pip install opencv-contrib-python")
    print("\nThen try running this script again.")
    sys.exit(1)

# ================ Configuration ================
API_PORT = 5050
ARDUINO_PORT = "/dev/ttyACM0"  # Common Arduino port on Linux, change as needed
ARDUINO_BAUD_RATE = 9600
CAMERA_ID = 0  # Default camera (0 is usually the built-in webcam)
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
RECOGNITION_CONFIDENCE_THRESHOLD = 50  # Lowered minimum confidence (0-100)
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Software", "mern", "client", "public", "photos")
MODEL_SAVE_PATH = "face_recognition_model.yml"  # Path to save/load trained model
MERN_SERVER_URL = "http://localhost:5050/record"  # URL of the MERN stack server
MONGODB_URI = "mongodb+srv://bn00017:QqqUP3%40duTjSxPu@cluster0.nh2ok3z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB = "employees"
MONGODB_COLLECTION = "patients"

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

# ================ MongoDB Connection ================
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DB]
    collection = db[MONGODB_COLLECTION]
    logger.info(f"Connected to MongoDB: {MONGODB_URI}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    mongo_client = None
    db = None
    collection = None

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

# ================ Helper Functions ================
def load_patient_data_from_mongodb():
    """Load patient data directly from MongoDB"""
    global patients
    
    if not mongo_client:
        logger.error("Cannot load patient data: MongoDB connection not available")
        return False
    
    try:
        # Get all patients from MongoDB
        patients_cursor = collection.find({})
        
        # Convert MongoDB cursor to list of patient objects
        patients = []
        for patient in patients_cursor:
            # Format the patient data
            patients.append({
                "id": str(patient["_id"]),
                "name": patient.get("name", "Unknown"),
                "photoUrl": patient.get("photoUrl", ""),
                "pillTimes": patient.get("pillTimes", ""),
                "slotNumber": int(patient.get("slotNumber", 0))
            })
        
        logger.info(f"Loaded {len(patients)} patients from MongoDB database")
        return True
    except Exception as e:
        logger.error(f"Error loading patient data from MongoDB: {e}")
        return False

def load_patient_data_from_api():
    """Load patient data from the MERN API as fallback"""
    global patients
    
    try:
        response = requests.get(f"{MERN_SERVER_URL}/arduino/patients")
        if response.status_code == 200:
            patients = response.json()
            logger.info(f"Loaded {len(patients)} patients from API")
            return True
        else:
            logger.error(f"Error fetching patient data: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to API: {e}")
        return False

def load_patient_photos():
    """Load and process patient photos for face recognition"""
    global patient_faces
    patient_faces = {}
    
    for patient in patients:
        patient_id = patient['id']
        photo_url = patient.get('photoUrl')
        
        if not photo_url:
            logger.warning(f"No photo specified for patient {patient['name']} ({patient_id})")
            continue
        
        # First, try to load photo from the specified path (could be a URL or file path)
        # In MongoDB, photoUrl could be a full URL, relative path, or base64 string
        
        # If it starts with http or https, download it
        if photo_url.startswith('http'):
            try:
                response = requests.get(photo_url)
                if response.status_code == 200:
                    # Save to local file
                    file_path = os.path.join(PHOTOS_DIR, f"{patient_id}.jpg")
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded photo for patient {patient['name']} ({patient_id}) from URL")
                else:
                    logger.warning(f"Failed to download photo for patient {patient['name']} ({patient_id}): HTTP {response.status_code}")
                    continue
            except Exception as e:
                logger.error(f"Error downloading photo for patient {patient['name']} ({patient_id}): {e}")
                continue
        # If it might be a base64 encoded image (starts with data:image)
        elif photo_url.startswith('data:image'):
            try:
                # Extract the base64 data
                encoded_data = photo_url.split(',')[1]
                image_data = base64.b64decode(encoded_data)
                
                # Save to local file
                file_path = os.path.join(PHOTOS_DIR, f"{patient_id}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"Decoded base64 photo for patient {patient['name']} ({patient_id})")
            except Exception as e:
                logger.error(f"Error decoding base64 photo for patient {patient['name']} ({patient_id}): {e}")
                continue
        # Otherwise, treat it as a file path
        else:
            # Check common locations
            possible_paths = [
                photo_url,
                os.path.join(PHOTOS_DIR, photo_url),
                os.path.join("public", photo_url),
                os.path.join("client", "public", photo_url),
                os.path.join("..", "Software", "mern", "client", "public", photo_url),
                os.path.join("..", "Software", "mern", "client", "src", "assets", photo_url)
            ]
            
            file_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            
            if not file_path:
                logger.warning(f"Photo file not found for patient {patient['name']} ({patient_id})")
                continue
        
        # Now file_path should be set to the local path of the photo
        try:
            # Load and process the photo
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
    global label_to_patient_map, recognizer
    
    if not patient_faces:
        logger.warning("No face data available to train the recognizer")
        return False
    
    faces = []
    labels = []
    label_to_patient_map = {}  # Map numeric labels to patient IDs
    
    # Create a new recognizer instance to reset previous training
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,           # Reduced radius for better performance
        neighbors=8,        # Standard number of neighbors
        grid_x=8,           # Increased grid for more accuracy
        grid_y=8,           # Increased grid for more accuracy
        threshold=100.0     # Higher threshold to avoid false negatives
    )
    
    next_label = 0
    for patient_id, face_list in patient_faces.items():
        label_to_patient_map[next_label] = patient_id
        logger.info(f"Training patient ID {patient_id} with label {next_label} - {len(face_list)} face samples")
        
        # Add multiple versions of each face with slight augmentation
        for face in face_list:
            # Original face
            resized_face = cv2.resize(face, (100, 100))
            faces.append(resized_face)
            labels.append(next_label)
            
            # Augment with small rotation variations
            rows, cols = resized_face.shape
            for angle in [-5, 5]:  # Small rotations
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                rotated = cv2.warpAffine(resized_face, M, (cols, rows))
                faces.append(rotated)
                labels.append(next_label)
                
            # Augment with brightness variations
            for factor in [0.9, 1.1]:  # Slight brightness changes
                brightened = cv2.convertScaleAbs(resized_face, alpha=factor, beta=0)
                faces.append(brightened)
                labels.append(next_label)
        
        next_label += 1
    
    if not faces:
        logger.warning("No valid faces to train the recognizer")
        return False
        
    recognizer.train(faces, np.array(labels))
    logger.info(f"Trained recognizer with {len(faces)} augmented faces from {len(patient_faces)} patients")
    
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

def check_patient_access(patient_id):
    """Check if the patient has access at the current time using MongoDB"""
    if mongo_client:
        try:
            # Get patient from MongoDB
            patient = collection.find_one({"_id": ObjectId(patient_id)})
            
            if not patient:
                logger.warning(f"Patient not found in MongoDB: {patient_id}")
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
                logger.info(f"No pill times specified for patient {patient['name']}, allowing access")
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
                        logger.info(f"Access granted for patient {patient['name']} (within {diff_minutes} minutes of {time_str})")
                        return True, f"Access granted (near pill time {time_str})", patient.get("slotNumber", 0)
                except Exception as e:
                    logger.error(f"Error parsing pill time '{time_str}': {e}")
            
            # No matching pill time found
            logger.info(f"Access denied for patient {patient['name']} (not within pill time window)")
            return False, "Access denied: Not within pill time window", patient.get("slotNumber", 0)
            
        except Exception as e:
            logger.error(f"Error checking patient access in MongoDB: {e}")
    
    # Fallback to API if MongoDB connection fails
    try:
        response = requests.get(f"{MERN_SERVER_URL}/arduino/access/{patient_id}")
        if response.status_code == 200:
            data = response.json()
            return data.get("access", False), data.get("message", ""), data.get("slotNumber", 0)
        else:
            logger.error(f"Error checking patient access via API: HTTP {response.status_code}")
            return False, "Error checking access", 0
    except Exception as e:
        logger.error(f"Error connecting to API for access check: {e}")
        return False, f"Connection error: {str(e)}", 0

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
    
    # Safety check: make sure we have a trained model
    if not patient_faces or not label_to_patient_map:
        logger.error("Cannot run face recognition - no trained model available")
        recognition_active = False
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        recognition_active = False
        return
    
    # Set resolution (lower for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    # Set FPS to improve performance
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    # Set camera buffer size to 1 to avoid lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set camera auto exposure for better image quality
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 is auto mode on many cameras
    
    last_recognition_time = 0
    recognition_cooldown = 1  # Reduced cooldown between recognitions
    
    logger.info("Starting face recognition thread")
    
    while recognition_active:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error reading from camera")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # Increased for faster detection
            minNeighbors=4,   # Reduced slightly for better detection
            minSize=(50, 50), # Larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        current_time = time.time()
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle for visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Check if enough time has passed since last recognition
            if current_time - last_recognition_time > recognition_cooldown:
                # Try to recognize the face
                # Pre-process the face for better recognition
                face_roi = gray[y:y+h, x:x+w]
                
                # Apply histogram equalization to improve contrast
                face_roi = cv2.equalizeHist(face_roi)
                
                # Resize to consistent size
                resized_face = cv2.resize(face_roi, (100, 100))
                
                try:
                    # Safety check to make sure the recognizer has been trained
                    if not label_to_patient_map:
                        logger.error("Face recognizer has not been trained!")
                        cv2.putText(frame, "Error: Model not trained", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        continue
                    
                    # Get multiple predictions with slightly different face crops for robustness
                    predictions = []
                    confidences = []
                    
                    # Original face
                    try:
                        label, confidence = recognizer.predict(resized_face)
                        predictions.append(label)
                        confidences.append(confidence)
                    except cv2.error as e:
                        if "This LBPH model is not computed yet" in str(e):
                            logger.error("Face recognizer model not trained properly!")
                            cv2.putText(frame, "Error: Model not trained", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            continue
                        else:
                            raise
                    
                    # Try different brightness variations for robustness
                    for factor in [0.85, 1.15]:
                        adjusted = cv2.convertScaleAbs(resized_face, alpha=factor, beta=0)
                        try:
                            label_bright, conf_bright = recognizer.predict(adjusted)
                            predictions.append(label_bright)
                            confidences.append(conf_bright)
                        except cv2.error:
                            # Skip if prediction fails
                            continue
                    
                    # Get the most common prediction if we have any predictions
                    if not predictions:
                        logger.warning("No valid predictions were made for this face")
                        cv2.putText(frame, "Could not recognize", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        continue
                        
                    from collections import Counter
                    pred_counts = Counter(predictions)
                    most_common_label = pred_counts.most_common(1)[0][0]
                    
                    # Get the best confidence for this label
                    best_confidence = min([conf for pred, conf in zip(predictions, confidences) if pred == most_common_label])
                    
                    # Convert confidence to a more intuitive 0-100 scale (100 is best match)
                    confidence_score = 100 - min(100, best_confidence)
                    
                    # Display confidence score on frame
                    cv2.putText(frame, f"Conf: {confidence_score:.1f}%", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Check if confidence meets threshold
                    if confidence_score >= RECOGNITION_CONFIDENCE_THRESHOLD:
                        if most_common_label in label_to_patient_map:
                            patient_id = label_to_patient_map[most_common_label]
                            
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
                            has_access, message, slot_number = check_patient_access(patient_id)
                            
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
                                send_command_to_arduino(f"ACCESS:{patient_id},{patient_name},{slot_number}")
                            else:
                                logger.info(f"Access denied: {message}")
                                send_command_to_arduino(f"DENY:{patient_id},{message}")
                                
                            last_recognition_time = current_time
                except Exception as e:
                    logger.error(f"Error during face recognition: {e}")
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        # More efficient way to control frame rate and reduce CPU usage
        # 30ms delay (~33 FPS), break on 'q' press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            recognition_active = False
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Face recognition thread stopped")

# ================ Initialize Flask API ================
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"status": "running", "message": "Face Recognition System"})

@app.route('/photos/<path:filename>')
def get_photo(filename):
    return send_from_directory(PHOTOS_DIR, filename)

@app.route('/status', methods=['GET'])
def get_status():
    status = {
        "arduino_connected": arduino is not None,
        "recognition_active": recognition_active,
        "patients_loaded": len(patients),
        "last_recognition": last_recognition_result,
        "mongodb_connected": mongo_client is not None
    }
    return jsonify(status)

@app.route('/start-recognition', methods=['POST'])
def start_recognition():
    global recognition_active, face_recognition_thread
    
    if recognition_active:
        return jsonify({"status": "already_running", "message": "Face recognition is already running"})
    
    recognition_active = True
    face_recognition_thread = threading.Thread(target=run_face_recognition)
    face_recognition_thread.daemon = True
    face_recognition_thread.start()
    
    return jsonify({"status": "started", "message": "Face recognition started"})

@app.route('/stop-recognition', methods=['POST'])
def stop_recognition():
    global recognition_active
    
    if not recognition_active:
        return jsonify({"status": "not_running", "message": "Face recognition is not running"})
    
    recognition_active = False
    # The thread will exit on its own when it checks the flag
    
    return jsonify({"status": "stopped", "message": "Face recognition stopping"})

@app.route('/reload-patients', methods=['POST'])
def reload_patients():
    """Reload patient data from MongoDB and retrain model"""
    global recognizer
    
    # Create new LBPH face recognizer with optimized parameters
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,           # Reduced radius for better performance
        neighbors=8,        # Standard number of neighbors
        grid_x=8,           # Increased grid for more accuracy
        grid_y=8,           # Increased grid for more accuracy
        threshold=100.0     # Higher threshold to avoid false negatives
    )
    
    # Load patient data
    if load_patient_data_from_mongodb() or load_patient_data_from_api():
        # Load patient photos and train the recognizer
        load_patient_photos()
        train_success = train_recognizer()
        
        if train_success:
            return jsonify({
                "status": "success", 
                "message": f"Reloaded {len(patients)} patients and retrained model with optimized parameters"
            })
        else:
            return jsonify({
                "status": "warning", 
                "message": f"Reloaded {len(patients)} patients but training failed - check photos and logs"
            }), 200
    else:
        return jsonify({"status": "error", "message": "Failed to reload patient data"}), 500

# ================ Main Function ================
def main():
    """Main function"""
    logger.info("Pill Dispenser Face Recognition System")
    logger.info("--------------------------------------")
    
    # Initialize optimized face recognizer
    global recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,           # Reduced radius for better performance
        neighbors=8,        # Standard number of neighbors
        grid_x=8,           # Increased grid for more accuracy
        grid_y=8,           # Increased grid for more accuracy
        threshold=100.0     # Higher threshold to avoid false negatives
    )
    
    # Load patient data (try MongoDB first, then API)
    if not load_patient_data_from_mongodb():
        logger.warning("Failed to load from MongoDB, trying API...")
        load_patient_data_from_api()
    
    # Always re-train the model for better accuracy
    logger.info("Loading patient photos and training face recognition model...")
    load_patient_photos()
    
    # Check if we have any patient faces to train with
    if not patient_faces:
        logger.error("No patient faces found for training! Cannot start face recognition.")
        logger.error("Please make sure patient photos are available and contain faces.")
        return
    
    # Train the recognizer
    training_success = train_recognizer()
    if not training_success:
        logger.error("Failed to train the face recognition model!")
        logger.error("Cannot proceed without a trained model.")
        return
        
    logger.info("Model training complete!")
    
    # Start the local web server in a separate thread
    server_thread = threading.Thread(target=app.run, kwargs={
        'host': '0.0.0.0',
        'port': 5051,  # Different port to avoid conflict with MERN
        'debug': False,
        'threaded': True  # Enable threading for better performance
    })
    server_thread.daemon = True
    server_thread.start()
    
    logger.info(f"Local API server running at http://localhost:5051")
    logger.info(f"Using MERN server at {MERN_SERVER_URL}")
    
    # Wait a moment before starting camera to ensure everything is initialized
    time.sleep(1)
    
    # Start face recognition only if training was successful
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
        
        # Clean up
        if arduino:
            arduino.close()
        
        if mongo_client:
            mongo_client.close()
        
        logger.info("Application terminated")

if __name__ == "__main__":
    main()