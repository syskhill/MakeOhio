#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import logging
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_recognition")

# Configuration
CAMERA_ID = 0
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
PHOTO_DIR = "/home/connor/MakeOhio/Software/mern/client/public/photos"

class FaceRecognitionTest:
    def __init__(self):
        # Initialize the face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)
        
        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=100.0
        )
        
        # Initialize camera
        self.camera = None
        self.is_running = False
        
        # Data structures
        self.patients = []
        self.patient_faces = {}
        self.label_to_patient_map = {}
    
    def initialize_camera(self):
        """Initialize the camera with optimized settings"""
        logger.info("Initializing camera...")
        self.camera = cv2.VideoCapture(CAMERA_ID)
        
        if not self.camera.isOpened():
            logger.error("Could not open camera!")
            return False
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.camera.set(cv2.CAP_PROP_FPS, 15)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to minimize lag
        
        logger.info("Camera initialized successfully")
        return True
    
    def load_test_faces(self):
        """Load test face photos from directory"""
        logger.info(f"Loading test faces from {PHOTO_DIR}")
        
        if not os.path.exists(PHOTO_DIR):
            logger.error(f"Photo directory {PHOTO_DIR} does not exist!")
            return False
        
        photo_files = [f for f in os.listdir(PHOTO_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not photo_files:
            logger.error("No photo files found!")
            return False
        
        logger.info(f"Found {len(photo_files)} test photos")
        
        for label_idx, photo_file in enumerate(photo_files):
            photo_path = os.path.join(PHOTO_DIR, photo_file)
            
            # Get patient name from filename (without extension)
            patient_name = os.path.splitext(photo_file)[0]
            
            # Process the photo
            img = cv2.imread(photo_path)
            if img is None:
                logger.warning(f"Could not read photo: {photo_path}")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better recognition
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            if len(faces) == 0:
                logger.warning(f"No face detected in {photo_file}")
                continue
            
            # Process the first face
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to a standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Create data augmentation versions
            augmented_faces = [face_roi]  # Start with original
            
            # Add brightness variations
            for factor in [0.9, 1.1]:
                brightened = cv2.convertScaleAbs(face_roi, alpha=factor, beta=0)
                augmented_faces.append(brightened)
            
            # Add rotation variations
            rows, cols = face_roi.shape
            for angle in [-5, 5]:
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                rotated = cv2.warpAffine(face_roi, M, (cols, rows))
                augmented_faces.append(rotated)
            
            # Store the face samples
            if patient_name not in self.patient_faces:
                self.patient_faces[patient_name] = []
                self.label_to_patient_map[label_idx] = patient_name
                self.patients.append({"name": patient_name, "label": label_idx})
            
            self.patient_faces[patient_name].extend(augmented_faces)
            logger.info(f"Added {len(augmented_faces)} samples for {patient_name}")
        
        return True
    
    def train_recognizer(self):
        """Train the face recognizer with loaded samples"""
        if not self.patient_faces:
            logger.error("No face samples to train with!")
            return False
        
        # Prepare training data
        faces = []
        labels = []
        
        for patient in self.patients:
            patient_name = patient["name"]
            label = patient["label"]
            
            for face in self.patient_faces[patient_name]:
                faces.append(face)
                labels.append(label)
        
        # Train the recognizer
        self.recognizer.train(faces, np.array(labels))
        logger.info(f"Trained recognizer with {len(faces)} samples from {len(self.patients)} patients")
        return True
    
    def run_camera_test(self, duration=30):
        """Run a camera test for the specified duration (in seconds)"""
        if not self.initialize_camera():
            return False
        
        logger.info(f"Running camera test for {duration} seconds...")
        self.is_running = True
        
        start_time = time.time()
        frame_count = 0
        recognition_count = 0
        
        # Lists to track predictions
        all_predictions = []
        correct_predictions = []
        
        while self.is_running and (time.time() - start_time) < duration:
            ret, frame = self.camera.read()
            
            if not ret:
                logger.error("Failed to grab frame!")
                break
            
            frame_count += 1
            
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=4,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Display camera info every 10 frames
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_time)
                logger.info(f"Camera FPS: {fps:.1f}, Faces detected: {len(faces)}")
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                # Only try to recognize if we have enough face area
                if w > 40 and h > 40:
                    # Multiple predictions for robustness
                    predictions = []
                    confidences = []
                    
                    # Test different versions of the face
                    for resize_size in [(100, 100)]:
                        resized = cv2.resize(face_roi, resize_size)
                        
                        # Original
                        label, confidence = self.recognizer.predict(resized)
                        predictions.append(label)
                        confidences.append(confidence)
                        
                        # Brightness variations
                        for factor in [0.85, 1.15]:
                            adjusted = cv2.convertScaleAbs(resized, alpha=factor, beta=0)
                            label_adj, conf_adj = self.recognizer.predict(adjusted)
                            predictions.append(label_adj)
                            confidences.append(conf_adj)
                    
                    # Get most common prediction
                    prediction_counts = Counter(predictions)
                    most_common_label = prediction_counts.most_common(1)[0][0]
                    
                    # Get best confidence for this label
                    best_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == most_common_label]
                    best_confidence = min(best_confidences) if best_confidences else float('inf')
                    
                    # Calculate confidence score (0-100, higher is better)
                    confidence_score = 100 - min(100, best_confidence)
                    
                    # Get predicted name
                    predicted_name = self.label_to_patient_map.get(most_common_label, "Unknown")
                    
                    # Display on frame
                    cv2.putText(frame, f"{predicted_name} ({confidence_score:.1f}%)", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    recognition_count += 1
                    all_predictions.append(predicted_name)
                    
                    # Log predictions every 5 recognitions
                    if recognition_count % 5 == 0:
                        logger.info(f"Prediction: {predicted_name} with confidence {confidence_score:.1f}%")
            
            # Display frame
            cv2.imshow('Face Recognition Test', frame)
            
            # Break on 'q' key
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"\nTest completed. Statistics:")
        logger.info(f"Elapsed time: {elapsed_time:.1f} seconds")
        logger.info(f"Frames processed: {frame_count}")
        logger.info(f"Average FPS: {fps:.1f}")
        logger.info(f"Total recognitions: {recognition_count}")
        
        # Print distribution of predictions
        if all_predictions:
            prediction_counts = Counter(all_predictions)
            logger.info("\nPrediction distribution:")
            for name, count in prediction_counts.most_common():
                percentage = (count / len(all_predictions)) * 100
                logger.info(f"{name}: {count} ({percentage:.1f}%)")
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Camera test completed")
        return True

def main():
    tester = FaceRecognitionTest()
    
    if tester.load_test_faces() and tester.train_recognizer():
        tester.run_camera_test(duration=60)  # Run test for 60 seconds
    else:
        logger.error("Failed to set up face recognition test!")

if __name__ == "__main__":
    main()
