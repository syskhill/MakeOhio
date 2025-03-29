#!/usr/bin/env python3

import cv2
import numpy as np
import os
import logging
import glob
import time
from collections import Counter
from pprint import pprint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("confidence_fix")

# Configuration
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Software", "mern", "client", "public", "photos")
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_SAVE_PATH = "face_recognition_model.yml"

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)

def check_photos_dir():
    """Check if photos directory exists and contains photos"""
    if not os.path.exists(PHOTOS_DIR):
        logger.error(f"Photos directory does not exist: {PHOTOS_DIR}")
        return False
    
    photos = glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.jpeg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.png"))
    
    if not photos:
        logger.error(f"No photos found in {PHOTOS_DIR}")
        return False
    
    logger.info(f"Found {len(photos)} photos in {PHOTOS_DIR}")
    return True

def train_model_v1():
    """Train the face recognition model using standard parameters"""
    # Create LBPH face recognizer with default parameters
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Load photos
    faces = []
    labels = []
    label_ids = {}
    next_label = 0
    
    photos = glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.jpeg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.png"))
    
    for photo_path in photos:
        patient_id = os.path.splitext(os.path.basename(photo_path))[0]
        logger.info(f"Processing photo for {patient_id}")
        
        # Assign label
        if patient_id not in label_ids:
            label_ids[patient_id] = next_label
            next_label += 1
        
        # Load and process photo
        img = cv2.imread(photo_path)
        if img is None:
            logger.warning(f"Could not read image: {photo_path}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(detected_faces) == 0:
            logger.warning(f"No face detected in {photo_path}")
            continue
        
        # Use the first face
        x, y, w, h = detected_faces[0]
        face_img = gray[y:y+h, x:x+w]
        
        # Resize to a standard size
        face_img = cv2.resize(face_img, (100, 100))
        
        # Add to training data
        faces.append(face_img)
        labels.append(label_ids[patient_id])
        
        logger.info(f"Added face for {patient_id} with label {label_ids[patient_id]}")
    
    # Train model if faces were found
    if len(faces) == 0:
        logger.error("No faces found in photos!")
        return None, None
    
    logger.info(f"Training model with {len(faces)} faces and {len(set(labels))} unique labels")
    recognizer.train(faces, np.array(labels))
    
    # Save model
    recognizer.save(MODEL_SAVE_PATH)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")
    
    return recognizer, {v: k for k, v in label_ids.items()}

def train_model_v2():
    """Train the face recognition model with optimized parameters"""
    # Create LBPH face recognizer with optimized parameters
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,           # Reduced radius for more localized features
        neighbors=8,        # Standard number of neighbors
        grid_x=8,           # Increased grid for more accuracy
        grid_y=8,           # Increased grid for more accuracy
        threshold=100.0     # Higher threshold to avoid false negatives
    )
    
    # Load photos
    faces = []
    labels = []
    label_ids = {}
    next_label = 0
    
    photos = glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.jpeg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.png"))
    
    for photo_path in photos:
        patient_id = os.path.splitext(os.path.basename(photo_path))[0]
        logger.info(f"Processing photo for {patient_id}")
        
        # Assign label
        if patient_id not in label_ids:
            label_ids[patient_id] = next_label
            next_label += 1
        
        # Load and process photo
        img = cv2.imread(photo_path)
        if img is None:
            logger.warning(f"Could not read image: {photo_path}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Detect faces
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(detected_faces) == 0:
            logger.warning(f"No face detected in {photo_path}")
            continue
        
        # Use the first face
        x, y, w, h = detected_faces[0]
        face_img = gray[y:y+h, x:x+w]
        
        # Resize to a standard size
        face_img = cv2.resize(face_img, (100, 100))
        
        # Apply Gaussian blur to reduce noise (optional)
        # face_img = cv2.GaussianBlur(face_img, (5, 5), 0)
        
        # Original image
        faces.append(face_img)
        labels.append(label_ids[patient_id])
        
        # Data augmentation - add slightly modified versions
        # Brightness variations
        for factor in [0.9, 1.1]:
            brightened = cv2.convertScaleAbs(face_img, alpha=factor, beta=0)
            faces.append(brightened)
            labels.append(label_ids[patient_id])
        
        # Rotation variations
        rows, cols = face_img.shape
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(face_img, M, (cols, rows))
            faces.append(rotated)
            labels.append(label_ids[patient_id])
        
        logger.info(f"Added face for {patient_id} with label {label_ids[patient_id]} and 4 augmented variations")
    
    # Train model if faces were found
    if len(faces) == 0:
        logger.error("No faces found in photos!")
        return None, None
    
    logger.info(f"Training model with {len(faces)} faces and {len(set(labels))} unique labels")
    recognizer.train(faces, np.array(labels))
    
    # Save model
    recognizer.save("face_recognition_model_v2.yml")
    logger.info(f"Model saved to face_recognition_model_v2.yml")
    
    return recognizer, {v: k for k, v in label_ids.items()}

def test_recognition(recognizer, label_map, version="v1"):
    """Test face recognition with the camera"""
    if recognizer is None or label_map is None:
        logger.error("Cannot test recognition - invalid model or label map")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    logger.info(f"Starting recognition test with model {version}")
    logger.info(f"Label map: {label_map}")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply same preprocessing as training
        if version == "v2":
            # CLAHE for v2
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        else:
            # Standard equalization for v1
            gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            
            # Recognize face
            try:
                label, confidence = recognizer.predict(face_img)
                
                # Convert confidence to percentage (LBPH: lower value = better match)
                # Different formula than typical - raw confidence for debugging
                confidence_pct = confidence
                
                # Get patient ID from label
                patient_id = label_map.get(label, "Unknown")
                
                # Display info
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {patient_id}", (x, y-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence_pct:.2f}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Log details for debugging
                logger.info(f"Recognition: ID={patient_id}, Label={label}, Raw Confidence={confidence}")
                
            except Exception as e:
                logger.error(f"Recognition error: {e}")
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display model version
        cv2.putText(frame, f"Model: {version}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow(f"Face Recognition Test - {version}", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    logger.info("Starting Face Recognition Confidence Fix Tool")
    
    # Check photos directory
    if not check_photos_dir():
        logger.error("Please add photos to the photos directory")
        logger.info(f"Expected location: {PHOTOS_DIR}")
        return
    
    # Train model with two different approaches for comparison
    recognizer1, label_map1 = train_model_v1()
    recognizer2, label_map2 = train_model_v2()
    
    if recognizer1 is None or recognizer2 is None:
        logger.error("Model training failed")
        return
    
    # Test recognition with both models
    logger.info("\nStarting test with standard model (v1)")
    logger.info("Press 'q' to close this test and continue to the next one")
    test_recognition(recognizer1, label_map1, "v1")
    
    logger.info("\nStarting test with optimized model (v2)")
    logger.info("Press 'q' to close this test and exit")
    test_recognition(recognizer2, label_map2, "v2")
    
    logger.info("Tests completed")

if __name__ == "__main__":
    main()
