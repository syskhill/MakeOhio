import cv2
import numpy as np
import time
import serial
import os
import json
import logging
from pymongo import MongoClient
from bson.objectid import ObjectId

# ================ Configuration ================
ARDUINO_PORT = "COM4"  # Use the correct COM port
ARDUINO_BAUD_RATE = 9600
CAMERA_ID = 0  # Default camera
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
MONGODB_URI = "mongodb+srv://bn00017:QqqUP3%40duTjSxPu@cluster0.nh2ok3z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB = "employees"
MONGODB_COLLECTION = "patients"

# Set up logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_recognition")

# ================ Connect to MongoDB ================
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DB]
    collection = db[MONGODB_COLLECTION]
    logger.info(f"Connected to MongoDB: {MONGODB_URI}")
    
    # Test connection by counting documents
    doc_count = collection.count_documents({})
    logger.info(f"Found {doc_count} patients in MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    mongo_client = None
    db = None
    collection = None

# ================ Connect to Arduino ================
arduino = None
try:
    arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
    logger.info(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2)  # Wait for Arduino to initialize
    
    # Send test command
    arduino.write(b"MESSAGE:System,Initializing\n")
    time.sleep(1)
    
    # Try to get response
    if arduino.in_waiting > 0:
        response = arduino.read(arduino.in_waiting).decode('utf-8')
        logger.info(f"Arduino response: {response}")
except Exception as e:
    logger.error(f"Error connecting to Arduino: {e}")
    arduino = None

# ================ Load Patient Data ================
patients = []

def load_patient_data():
    """Load patient data from MongoDB"""
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
                "slotNumber": int(patient.get("slotNumber", 0))
            })
        
        logger.info(f"Loaded {len(patients)} patients from MongoDB database")
        
        # Print patient details for debugging
        for i, patient in enumerate(patients):
            logger.info(f"Patient {i+1}: ID={patient['id']}, Name={patient['name']}, Slot={patient['slotNumber']}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading patient data from MongoDB: {e}")
        return False

# ================ Run Face Detection ================
def run_face_detection():
    """Run face detection with camera"""
    if not load_patient_data():
        logger.error("Failed to load patient data, cannot continue")
        return
    
    # Let Arduino know we're starting face detection
    if arduino:
        arduino.write(b"MESSAGE:Face Detection,Starting\n")
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)
    logger.info("Loaded face detection model")
    
    # Start camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        logger.error("Could not open camera")
        return
    
    logger.info("Camera opened successfully")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Main detection loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame from camera")
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process detected faces
            if len(faces) > 0:
                logger.info(f"Detected {len(faces)} faces in frame")
                
                # For simplicity, just use the first face and first patient
                # In a real system, you would compare with known face encodings
                if len(patients) > 0:
                    patient = patients[0]
                    
                    # Draw rectangle around face
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Send command to Arduino
                    if arduino:
                        command = f"ACCESS:{patient['id']},{patient['name']},{patient['slotNumber']}\n"
                        logger.info(f"Sending command to Arduino: {command.strip()}")
                        arduino.write(command.encode())
                        
                        # Wait for response
                        time.sleep(2)
                        if arduino.in_waiting > 0:
                            response = arduino.read(arduino.in_waiting).decode('utf-8')
                            logger.info(f"Arduino response: {response}")
                    
                    # Display patient info on screen
                    cv2.putText(frame, f"Name: {patient['name']}", 
                                (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Access Granted", 
                                (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Wait before next detection
                    time.sleep(5)
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        logger.error(f"Error in face detection loop: {e}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        if arduino:
            arduino.write(b"MESSAGE:System,Shutting Down\n")
            arduino.close()
        
        if mongo_client:
            mongo_client.close()
        
        logger.info("Face recognition system shut down")

if __name__ == "__main__":
    logger.info("Starting Face Recognition System")
    logger.info("------------------------------")
    run_face_detection()