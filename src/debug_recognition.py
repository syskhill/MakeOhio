#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
import glob
import logging
import sys

# Configure logging with timestamp
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Force output to stdout
    ]
)
logger = logging.getLogger("debug_recognition")

# Make sure logger is set to output everything
logger.setLevel(logging.DEBUG)

# Configuration
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Software", "mern", "client", "public", "photos")
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_SAVE_PATH = "debug_model.yml"

def print_banner(message):
    """Print a banner with a message"""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80 + "\n")

def check_opencv_setup():
    """Check OpenCV setup and version"""
    print_banner("OpenCV Setup Check")
    
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check if face module is available
    face_modules_available = hasattr(cv2, 'face')
    print(f"Face module available: {face_modules_available}")
    
    if not face_modules_available:
        logger.error("OpenCV face module not available. Install opencv-contrib-python")
        print("Run: pip install opencv-contrib-python")
        return False
    
    # Check for face cascade file
    cascade_path = cv2.data.haarcascades + FACE_CASCADE_PATH
    cascade_exists = os.path.exists(cascade_path)
    print(f"Face cascade file exists: {cascade_exists}")
    print(f"Cascade path: {cascade_path}")
    
    if not cascade_exists:
        logger.error(f"Haar cascade file not found at {cascade_path}")
        return False
    
    print("\nOpenCV setup looks good!")
    return True

def check_photos_directory():
    """Check the photos directory"""
    print_banner("Photos Directory Check")
    
    # Check if directory exists
    dir_exists = os.path.exists(PHOTOS_DIR)
    print(f"Photos directory exists: {dir_exists}")
    print(f"Photos directory path: {PHOTOS_DIR}")
    
    if not dir_exists:
        logger.error(f"Photos directory does not exist: {PHOTOS_DIR}")
        print(f"Creating directory: {PHOTOS_DIR}")
        os.makedirs(PHOTOS_DIR, exist_ok=True)
    
    # Check for photos
    photos = glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.jpeg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.png"))
    
    print(f"Number of photos found: {len(photos)}")
    
    if len(photos) == 0:
        logger.warning("No photos found in photos directory")
        print(f"Please add photos to: {PHOTOS_DIR}")
        return False
    
    # Print list of photos
    print("\nPhotos found:")
    for i, photo in enumerate(photos[:10]):  # Show first 10
        print(f"  {i+1}. {os.path.basename(photo)}")
    
    if len(photos) > 10:
        print(f"  ... and {len(photos) - 10} more")
    
    return True

def train_recognition_model():
    """Train face recognition model with debug output"""
    print_banner("Training Face Recognition Model")
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)
    logger.info("Face cascade classifier initialized")
    
    # Initialize face recognition with standard parameters
    # Using the exact same parameters as the main system
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    logger.info("Face recognizer created with DEFAULT parameters")
    
    # Get list of photos
    photos = glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.jpeg")) + \
             glob.glob(os.path.join(PHOTOS_DIR, "*.png"))
    
    if len(photos) == 0:
        logger.error("No photos found for training")
        return None, None
    
    # Process photos for training
    faces = []
    labels = []
    label_map = {}
    next_label = 0
    
    print("\nProcessing photos for training:")
    for i, photo_path in enumerate(photos):
        # Get patient ID from filename
        patient_id = os.path.splitext(os.path.basename(photo_path))[0]
        print(f"\nPhoto {i+1}/{len(photos)}: {patient_id}")
        
        # Assign numeric label
        if patient_id not in label_map:
            label_map[patient_id] = next_label
            next_label += 1
        
        # Load image
        print(f"  Loading image from: {photo_path}")
        img = cv2.imread(photo_path)
        
        if img is None:
            print(f"  ERROR: Could not read image file")
            continue
        
        print(f"  Image loaded successfully: {img.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"  Converted to grayscale: {gray.shape}")
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        print(f"  Applied CLAHE enhancement")
        
        # Detect face
        print(f"  Detecting faces...")
        detected_faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        print(f"  Detected {len(detected_faces)} faces")
        
        if len(detected_faces) == 0:
            print(f"  WARNING: No faces detected in this image")
            # Save debug image
            debug_path = f"debug_{patient_id}_noface.jpg"
            cv2.imwrite(debug_path, img)
            print(f"  Debug image saved to: {debug_path}")
            continue
        
        # Process the first face
        x, y, w, h = detected_faces[0]
        print(f"  Using face at position: x={x}, y={y}, width={w}, height={h}")
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        print(f"  Extracted face ROI: {face_roi.shape}")
        
        # Resize to standard size
        face_resized = cv2.resize(face_roi, (100, 100))
        print(f"  Resized to: {face_resized.shape}")
        
        # Save face image for debugging
        face_debug_path = f"debug_{patient_id}_face.jpg"
        cv2.imwrite(face_debug_path, face_resized)
        print(f"  Debug face image saved to: {face_debug_path}")
        
        # Add to training data
        faces.append(face_resized)
        labels.append(label_map[patient_id])
        print(f"  Added to training data with label: {label_map[patient_id]}")
        
        # Create augmented versions
        print(f"  Creating augmented versions...")
        
        # Brightness variations
        for factor in [0.9, 1.1]:
            brightened = cv2.convertScaleAbs(face_resized, alpha=factor, beta=0)
            faces.append(brightened)
            labels.append(label_map[patient_id])
        
        # Rotation variations
        rows, cols = face_resized.shape
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(face_resized, M, (cols, rows))
            faces.append(rotated)
            labels.append(label_map[patient_id])
        
        print(f"  Added 4 augmented versions")
    
    # Train model
    if len(faces) == 0:
        print("\nERROR: No valid faces found for training")
        return None, None
    
    print(f"\nTraining model with {len(faces)} face images and {len(set(labels))} unique people")
    recognizer.train(faces, np.array(labels))
    print("Model training completed")
    
    # Save model
    recognizer.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Create reverse mapping
    id_to_label = {v: k for k, v in label_map.items()}
    print(f"Label mapping: {id_to_label}")
    
    return recognizer, id_to_label

def test_recognition(recognizer, label_map):
    """Test face recognition with camera"""
    print_banner("Face Recognition Test")
    
    if recognizer is None or label_map is None:
        logger.error("Cannot test - model not trained or label map not available")
        return
    
    print(f"Label map: {label_map}")
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)
    print("Face cascade classifier initialized")
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    print("Camera opened successfully")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag
    
    # Variables for FPS tracking
    frame_count = 0
    start_time = time.time()
    last_log_time = start_time
    
    print("\nStarting recognition loop - press 'q' to quit")
    print("Debug information will be printed to console\n")
    
    # Main loop
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Failed to read frame from camera")
            break
        
        # Update stats
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Log stats every second
        if current_time - last_log_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}, Frames processed: {frame_count}")
            last_log_time = current_time
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # Process each face
        if len(faces) > 0:
            print(f"Frame {frame_count}: Detected {len(faces)} faces")
            
            for i, (x, y, w, h) in enumerate(faces):
                print(f"  Face {i+1}: position=({x},{y}), size=({w},{h})")
                
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (100, 100))
                
                # Try recognition
                try:
                    # Get raw prediction
                    label, confidence = recognizer.predict(face_resized)
                    print(f"  Raw prediction: label={label}, distance={confidence}")
                    
                    # Get ID from label
                    person_id = label_map.get(label, "Unknown")
                    print(f"  Person ID: {person_id}")
                    
                    # Convert confidence to percentage
                    # LBPH returns distance values (lower is better)
                    # Try both conversion formulas for comparison
                    
                    # Formula 1: Original from face_recognition_fix.py
                    confidence_pct1 = max(0, min(100, 100 * (1 - (confidence / 100))))
                    
                    # Formula 2: Simple direct conversion (100 - distance)
                    confidence_pct2 = max(0, 100 - min(100, confidence))
                    
                    # Formula 3: Non-linear normalization
                    confidence_pct3 = max(0, min(100, 100 * np.exp(-0.01 * confidence)))
                    
                    # Use Formula 2 for display - seems most reliable
                    confidence_pct = confidence_pct2
                    
                    print(f"  Confidence calculations:")
                    print(f"    - Formula 1 (1-(d/100)): {confidence_pct1:.2f}%")
                    print(f"    - Formula 2 (100-d): {confidence_pct2:.2f}%")
                    print(f"    - Formula 3 (exp): {confidence_pct3:.2f}%")
                    print(f"  Confidence: {confidence_pct:.2f}%")
                    
                    # Display on frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {person_id}", (x, y-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {confidence_pct:.1f}% (Dist: {confidence:.1f})", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"  ERROR during recognition: {e}")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Recognition Error", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # No faces detected this frame
            pass
        
        # Show FPS
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow("Debug Face Recognition", frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nRecognition test completed")

def main():
    """Main function"""
    print_banner("Face Recognition Debug Tool")
    print("This tool will help diagnose face recognition issues")
    
    # Check OpenCV setup
    if not check_opencv_setup():
        print("OpenCV setup check failed. Please fix issues before continuing.")
        return
    
    # Check photos directory
    if not check_photos_directory():
        print("Photos directory check failed. Please add photos before continuing.")
        return
    
    # Train model
    recognizer, label_map = train_recognition_model()
    
    if recognizer is None:
        print("Model training failed. Cannot continue to testing.")
        return
    
    # Test recognition
    test_recognition(recognizer, label_map)
    
    print("\nDebug process completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"\nERROR: {e}")
        print("An unexpected error occurred. See above for details.")