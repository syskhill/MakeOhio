#!/usr/bin/env python3
"""
Face Recognition Model Trainer

This script handles training the face recognition model separately from recognition.
It loads patient photos, trains the LBPH face recognizer model, and saves the model
and patient mapping to disk so that face_recognition.py can use them without retraining.
"""

import cv2
import numpy as np
import os
import sys
import json
import time
import logging
import argparse
import glob
from pymongo import MongoClient
from bson.objectid import ObjectId
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_face_model")

# ================ Configuration ================
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_SAVE_PATH = "face_recognition_model.yml"
MAPPING_SAVE_PATH = "patient_mapping.pkl"
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Software", "mern", "client", "public", "photos")
TEST_PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_faces")
MONGODB_URI = "mongodb+srv://bn00017:QqqUP3%40duTjSxPu@cluster0.nh2ok3z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB = "employees"
MONGODB_COLLECTION = "patients"

# ================ Helper Functions ================
def connect_to_mongodb():
    """Connect to MongoDB database"""
    try:
        logger.info(f"Connecting to MongoDB: {MONGODB_URI}")
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]
        logger.info(f"Connected to MongoDB database: {MONGODB_DB}.{MONGODB_COLLECTION}")
        return mongo_client, db, collection
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None, None, None

def load_patient_data_from_mongodb(collection):
    """Load patient data from MongoDB"""
    if not collection:
        logger.error("Cannot load patient data: MongoDB connection not available")
        return []
    
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
        return patients
    except Exception as e:
        logger.error(f"Error loading patient data from MongoDB: {e}")
        return []

def load_patient_data_from_directory(photos_dir):
    """Load patient data from photo directory without MongoDB"""
    try:
        patients = []
        # Get all image files in the directory
        image_files = glob.glob(os.path.join(photos_dir, "*.jpg")) + \
                     glob.glob(os.path.join(photos_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(photos_dir, "*.png"))
        
        # Extract patient IDs from filenames
        patient_ids = set()
        for image_path in image_files:
            filename = os.path.basename(image_path)
            # Handle multi-photo format (patient_id_N.jpg)
            if "_" in filename:
                parts = filename.split("_")
                patient_id = parts[0]  # Extract ID part
            else:
                # Handle single photo format (patient_id.jpg)
                patient_id = os.path.splitext(filename)[0]
            
            patient_ids.add(patient_id)
        
        # Create patient records for each ID
        for patient_id in patient_ids:
            patients.append({
                "id": patient_id,
                "name": f"Patient {patient_id}",  # Default name
                "photoUrl": "",  # Will be filled during photo loading
                "pillTimes": "8:00,12:00,18:00",  # Default times
                "slotNumber": 1  # Default slot
            })
        
        logger.info(f"Loaded {len(patients)} patients from directory {photos_dir}")
        return patients
    except Exception as e:
        logger.error(f"Error loading patient data from directory: {e}")
        return []

def load_patient_photos(patients, photos_dir, face_cascade):
    """Load and process patient photos for face recognition"""
    if not os.path.exists(photos_dir):
        logger.error(f"Photos directory not found: {photos_dir}")
        os.makedirs(photos_dir, exist_ok=True)
        logger.info(f"Created photos directory: {photos_dir}")
    
    patient_faces = {}  # Dictionary to store face images by patient ID
    
    # First, check for multi-photo format (patient_id_1.jpg, patient_id_2.jpg, etc.)
    for patient in patients:
        patient_id = patient['id']
        multi_photos = []
        
        # Look for multiple photos with the pattern patient_id_X.jpg
        for ext in [".jpg", ".jpeg", ".png"]:
            pattern = os.path.join(photos_dir, f"{patient_id}_*{ext}")
            multi_photos.extend(glob.glob(pattern))
        
        # If we found multiple photos, process them
        if multi_photos:
            logger.info(f"Found {len(multi_photos)} photos for patient {patient['name']} ({patient_id})")
            patient_faces[patient_id] = []
            
            for photo_path in multi_photos:
                try:
                    # Update the photoUrl in the patient record for the first photo
                    if not patient['photoUrl']:
                        patient['photoUrl'] = os.path.basename(photo_path)
                    
                    # Load and process the photo
                    img = cv2.imread(photo_path)
                    if img is None:
                        logger.warning(f"Failed to read image file: {photo_path}")
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]  # Take the first face found
                        face_img = gray[y:y+h, x:x+w]
                        patient_faces[patient_id].append(face_img)
                        logger.info(f"Loaded face from {os.path.basename(photo_path)} for {patient['name']}")
                    else:
                        logger.warning(f"No face detected in {os.path.basename(photo_path)}")
                except Exception as e:
                    logger.error(f"Error processing photo {os.path.basename(photo_path)}: {e}")
            
            # If we processed at least one face successfully, continue
            if patient_faces[patient_id]:
                continue
        
        # If no multi-photos or they all failed, try the regular photoUrl or ID-based filename
        # Try different filename patterns
        potential_files = [
            os.path.join(photos_dir, f"{patient_id}.jpg"),
            os.path.join(photos_dir, f"{patient_id}.jpeg"),
            os.path.join(photos_dir, f"{patient_id}.png"),
        ]
        
        # Add photoUrl if available
        if patient.get('photoUrl'):
            potential_files.append(os.path.join(photos_dir, patient['photoUrl']))
        
        # Try each potential file
        for file_path in potential_files:
            if os.path.exists(file_path):
                try:
                    if not patient['photoUrl']:
                        patient['photoUrl'] = os.path.basename(file_path)
                        
                    # Load and process the photo
                    img = cv2.imread(file_path)
                    if img is None:
                        logger.warning(f"Failed to read image file: {file_path}")
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]  # Take the first face found
                        face_img = gray[y:y+h, x:x+w]
                        
                        # Initialize the face list if needed
                        if patient_id not in patient_faces:
                            patient_faces[patient_id] = []
                        
                        patient_faces[patient_id].append(face_img)
                        logger.info(f"Loaded face for {patient['name']} ({patient_id}) from {os.path.basename(file_path)}")
                        break  # Stop after first successful image
                    else:
                        logger.warning(f"No face detected in photo for {patient['name']} ({patient_id})")
                except Exception as e:
                    logger.error(f"Error processing photo for {patient['name']} ({patient_id}): {e}")
    
    logger.info(f"Processed photos for {len(patient_faces)} patients with valid faces")
    return patient_faces, patients

def train_recognizer(patient_faces):
    """Train the face recognizer with loaded patient faces"""
    if not patient_faces:
        logger.warning("No face data available to train the recognizer")
        return None, {}
    
    faces = []
    labels = []
    label_to_patient_map = {}  # Map numeric labels to patient IDs
    
    # Create a new recognizer instance with DEFAULT parameters
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
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
        return None, {}
        
    # Train the model
    try:
        recognizer.train(faces, np.array(labels))
        logger.info(f"Trained recognizer with {len(faces)} augmented faces from {len(patient_faces)} patients")
        return recognizer, label_to_patient_map
    except Exception as e:
        logger.error(f"Error training recognizer: {e}")
        return None, {}

def save_model_and_mapping(recognizer, label_to_patient_map, model_path, mapping_path):
    """Save the trained model and patient mapping to disk"""
    if not recognizer:
        logger.error("No recognizer to save")
        return False
    
    try:
        # Save the trained model
        recognizer.save(model_path)
        logger.info(f"Saved face recognition model to {model_path}")
        
        # Save the label to patient mapping
        with open(mapping_path, 'wb') as f:
            pickle.dump(label_to_patient_map, f)
        logger.info(f"Saved patient mapping to {mapping_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving model or mapping: {e}")
        return False

def verify_model(recognizer, patient_faces, label_to_patient_map, patients):
    """Verify the trained model by testing against training data"""
    if not recognizer or not patient_faces:
        logger.error("No recognizer or patient faces to verify")
        return
    
    logger.info("Verifying trained model against training data...")
    
    # Create a reverse mapping from patient ID to name
    id_to_name = {patient["id"]: patient["name"] for patient in patients}
    
    total_tests = 0
    correct_recognitions = 0
    
    # Test the model on each face
    for true_label, patient_id in label_to_patient_map.items():
        if patient_id not in patient_faces:
            continue
            
        face_list = patient_faces[patient_id]
        patient_name = id_to_name.get(patient_id, f"Patient {patient_id}")
        
        for i, face in enumerate(face_list):
            total_tests += 1
            
            # Resize and preprocess
            resized_face = cv2.resize(face, (100, 100))
            
            # Predict
            try:
                label, confidence = recognizer.predict(resized_face)
                confidence_score = max(0, 100 - min(100, confidence))
                
                predicted_patient_id = label_to_patient_map.get(label, "Unknown")
                predicted_patient_name = id_to_name.get(predicted_patient_id, f"Patient {predicted_patient_id}")
                
                if label == true_label:
                    correct_recognitions += 1
                    logger.info(f"✅ Correct: {patient_name} recognized with {confidence_score:.1f}% confidence")
                else:
                    logger.info(f"❌ Error: {patient_name} recognized as {predicted_patient_name} with {confidence_score:.1f}% confidence")
            except Exception as e:
                logger.error(f"Error predicting face for {patient_name}: {e}")
    
    # Calculate accuracy
    if total_tests > 0:
        accuracy = (correct_recognitions / total_tests) * 100
        logger.info(f"Model verification complete - Accuracy: {accuracy:.1f}% ({correct_recognitions}/{total_tests})")
    else:
        logger.warning("No tests performed during verification")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Face Recognition Model Trainer")
    parser.add_argument("--no-mongodb", action="store_true", help="Skip MongoDB and load patients from photo directory")
    parser.add_argument("--photos-dir", help="Directory containing patient photos (default: photos)")
    parser.add_argument("--model-path", help="Path to save trained model (default: face_recognition_model.yml)")
    parser.add_argument("--mapping-path", help="Path to save patient mapping (default: patient_mapping.pkl)")
    parser.add_argument("--verify", action="store_true", help="Verify the model after training")
    parser.add_argument("--test-dir", help="Directory containing test photos")
    args = parser.parse_args()
    
    # Update paths from arguments if provided
    global PHOTOS_DIR, MODEL_SAVE_PATH, MAPPING_SAVE_PATH, TEST_PHOTOS_DIR
    
    if args.photos_dir:
        PHOTOS_DIR = args.photos_dir
    if args.model_path:
        MODEL_SAVE_PATH = args.model_path
    if args.mapping_path:
        MAPPING_SAVE_PATH = args.mapping_path
    if args.test_dir:
        TEST_PHOTOS_DIR = args.test_dir
    
    logger.info("Face Recognition Model Trainer")
    logger.info("=============================")
    logger.info(f"Photos directory: {PHOTOS_DIR}")
    logger.info(f"Model save path: {MODEL_SAVE_PATH}")
    logger.info(f"Mapping save path: {MAPPING_SAVE_PATH}")
    
    # Make sure photos directory exists
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    
    # Initialize OpenCV face modules
    try:
        cascade_path = cv2.data.haarcascades + FACE_CASCADE_PATH
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error(f"Failed to load face cascade classifier")
            return
        
        # Initialize face recognizer to verify it works
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        logger.info("OpenCV face modules initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing OpenCV face modules: {e}")
        logger.error("Make sure opencv-contrib-python is installed (pip install opencv-contrib-python)")
        return
    
    # Load patient data
    patients = []
    if args.no_mongodb:
        logger.info("Skipping MongoDB - loading patients from photo directory")
        patients = load_patient_data_from_directory(PHOTOS_DIR)
    else:
        # Connect to MongoDB
        mongo_client, db, collection = connect_to_mongodb()
        if collection:
            patients = load_patient_data_from_mongodb(collection)
        else:
            logger.warning("MongoDB connection failed - falling back to directory-based loading")
            patients = load_patient_data_from_directory(PHOTOS_DIR)
    
    if not patients:
        logger.error("No patients loaded. Cannot continue.")
        return
    
    # Load patient photos
    patient_faces, updated_patients = load_patient_photos(patients, PHOTOS_DIR, face_cascade)
    
    if not patient_faces:
        logger.error("No valid patient faces loaded. Cannot train model.")
        return
    
    # Train the recognizer
    recognizer, label_to_patient_map = train_recognizer(patient_faces)
    
    if not recognizer:
        logger.error("Failed to train face recognizer. Cannot save model.")
        return
    
    # Save model and mapping
    if save_model_and_mapping(recognizer, label_to_patient_map, MODEL_SAVE_PATH, MAPPING_SAVE_PATH):
        logger.info("Model and mapping saved successfully")
        
        # Verify the model if requested
        if args.verify:
            verify_model(recognizer, patient_faces, label_to_patient_map, updated_patients)
    else:
        logger.error("Failed to save model and mapping")
    
    logger.info("Training complete")

if __name__ == "__main__":
    main()