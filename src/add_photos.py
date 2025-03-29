#!/usr/bin/env python3

import os
import sys
import cv2
import time
import shutil
import logging
import argparse
import uuid
from pymongo import MongoClient
from bson.objectid import ObjectId

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("add_photos")

# MongoDB connection information
MONGODB_URI = "mongodb+srv://bn00017:QqqUP3%40duTjSxPu@cluster0.nh2ok3z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB = "employees"
MONGODB_COLLECTION = "patients"

# Photo directory
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Software", "mern", "client", "public", "photos")

def connect_mongodb():
    """Connect to MongoDB and return collection"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]
        return client, collection
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None, None

def list_patients(collection):
    """List all patients in the database"""
    try:
        patients = list(collection.find({}))
        if not patients:
            print("No patients found in the database.")
            return []
        
        print("\n===== Available Patients =====")
        for i, patient in enumerate(patients):
            patient_id = str(patient['_id'])
            name = patient.get('name', 'Unknown')
            
            # Count existing photos for this patient
            photo_count = count_patient_photos(patient_id)
            
            print(f"{i+1}. {name} (ID: {patient_id}) - {photo_count} photo(s)")
            
        return patients
    except Exception as e:
        logger.error(f"Error listing patients: {e}")
        return []

def count_patient_photos(patient_id):
    """Count how many photos exist for a patient"""
    count = 0
    for file in os.listdir(PHOTOS_DIR):
        if file.startswith(f"{patient_id}_") and file.endswith((".jpg", ".jpeg", ".png")):
            count += 1
    return count

def capture_photo(patient_id, patient_name, index=None):
    """Capture photo from webcam"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return None
    
    # Set up window
    window_name = f"Capture Photo for {patient_name}"
    cv2.namedWindow(window_name)
    
    # Instructions
    print("\nCapturing photo from webcam:")
    print("1. Position face in the center of the frame")
    print("2. Press SPACE to capture")
    print("3. Press ESC to cancel")
    
    # Face detection for guidance
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            break
        
        # Face detection for visual guidance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add instructions to the frame
        cv2.putText(frame, "Press SPACE to capture, ESC to cancel", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show how many faces are detected
        cv2.putText(frame, f"Detected: {len(faces)} face(s)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the current frame
        cv2.imshow(window_name, frame)
        
        # Check for key press
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            print("Cancelled photo capture")
            break
        elif key == 32:  # SPACE key
            # Only capture if exactly one face is detected
            if len(faces) == 1:
                # Generate filename
                if index is None:
                    # Get the current count and add 1
                    index = count_patient_photos(patient_id) + 1
                
                # Create filename with patient ID and index
                filename = f"{patient_id}_{index}.jpg"
                file_path = os.path.join(PHOTOS_DIR, filename)
                
                # Save the image
                cv2.imwrite(file_path, frame)
                print(f"Photo saved to {file_path}")
                
                # Show confirmation
                cv2.putText(frame, "Photo Captured!", 
                           (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1000)  # Show confirmation for 1 second
                
                # Return the path of the saved image
                cap.release()
                cv2.destroyAllWindows()
                return file_path
            else:
                # Show warning if no face or multiple faces detected
                message = "No face detected" if len(faces) == 0 else "Multiple faces detected"
                cv2.putText(frame, message, 
                           (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1000)  # Show warning for 1 second
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    return None

def import_photo(patient_id, index=None):
    """Import a photo from file system"""
    import tkinter as tk
    from tkinter import filedialog
    
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Photo",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if not file_path:
        print("No file selected")
        return None
    
    try:
        # Read image with OpenCV to verify it's valid
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"Could not read image file: {file_path}")
            return None
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            logger.warning("No faces detected in the selected image")
            proceed = input("No faces detected. Import anyway? (y/n): ").lower()
            if proceed != 'y':
                return None
        elif len(faces) > 1:
            logger.warning(f"Multiple faces ({len(faces)}) detected in the selected image")
            proceed = input("Multiple faces detected. Import anyway? (y/n): ").lower()
            if proceed != 'y':
                return None
        
        # Generate filename
        if index is None:
            # Get the current count and add 1
            index = count_patient_photos(patient_id) + 1
        
        # Create filename with patient ID and index
        filename = f"{patient_id}_{index}.jpg"
        dest_path = os.path.join(PHOTOS_DIR, filename)
        
        # Copy the file
        shutil.copy2(file_path, dest_path)
        print(f"Photo imported to {dest_path}")
        
        return dest_path
    except Exception as e:
        logger.error(f"Error importing photo: {e}")
        return None

def update_patient_photo(collection, patient_id, photo_path):
    """Update patient's primary photo URL in MongoDB"""
    try:
        # Extract filename from path
        filename = os.path.basename(photo_path)
        
        # Update MongoDB record
        result = collection.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": {"photoUrl": filename}}
        )
        
        if result.modified_count > 0:
            print(f"Updated primary photo for patient ID: {patient_id}")
            return True
        else:
            logger.warning(f"No changes made to patient record")
            return False
    except Exception as e:
        logger.error(f"Error updating patient record: {e}")
        return False

def add_new_patient(collection, name):
    """Add a new patient to the database"""
    try:
        # Get the next available slot number
        existing_slots = [p.get('slotNumber', 0) for p in collection.find({}, {"slotNumber": 1})]
        next_slot = 1
        while next_slot in existing_slots:
            next_slot += 1
        
        # Create new patient record
        new_patient = {
            "name": name,
            "photoUrl": "",  # Will be updated after photo is taken
            "pillTimes": "8:00,12:00,18:00",  # Default pill times
            "slotNumber": next_slot
        }
        
        # Insert into database
        result = collection.insert_one(new_patient)
        patient_id = str(result.inserted_id)
        
        print(f"Added new patient: {name} (ID: {patient_id}, Slot: {next_slot})")
        return patient_id
    except Exception as e:
        logger.error(f"Error adding new patient: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Add or update patient photos for face recognition")
    parser.add_argument("--new", action="store_true", help="Add a new patient")
    parser.add_argument("--import", dest="import_file", action="store_true", help="Import photo from file instead of webcam")
    args = parser.parse_args()
    
    # Make sure photos directory exists
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    
    # Connect to MongoDB
    mongo_client, collection = connect_mongodb()
    if not collection:
        print("Could not connect to MongoDB. Exiting.")
        return
    
    try:
        if args.new:
            # Add a new patient
            name = input("Enter patient name: ")
            if not name:
                print("Name cannot be empty")
                return
                
            # Add patient to database
            patient_id = add_new_patient(collection, name)
            if not patient_id:
                print("Failed to add new patient")
                return
                
            # Capture or import photo
            if args.import_file:
                photo_path = import_photo(patient_id, 1)
            else:
                photo_path = capture_photo(patient_id, name, 1)
                
            if photo_path:
                # Update patient record with photo URL
                update_patient_photo(collection, patient_id, photo_path)
            else:
                print("No photo added. Patient created without photo.")
        else:
            # List existing patients
            patients = list_patients(collection)
            if not patients:
                return
                
            # Let user select a patient
            selection = input("\nSelect patient number (or 'q' to quit): ")
            if selection.lower() == 'q':
                return
                
            try:
                index = int(selection) - 1
                if index < 0 or index >= len(patients):
                    print("Invalid selection")
                    return
                    
                patient = patients[index]
                patient_id = str(patient['_id'])
                name = patient.get('name', 'Unknown')
                
                print(f"\nSelected: {name} (ID: {patient_id})")
                
                # Ask to capture a photo or import
                if args.import_file:
                    photo_path = import_photo(patient_id)
                else:
                    photo_path = capture_photo(patient_id, name)
                
                if photo_path:
                    # Ask if this should be the primary photo
                    make_primary = input("Make this the primary photo for face recognition? (y/n): ").lower()
                    if make_primary == 'y':
                        update_patient_photo(collection, patient_id, photo_path)
                else:
                    print("No photo added.")
                
            except ValueError:
                print("Invalid input. Please enter a number.")
    finally:
        # Close MongoDB connection
        if mongo_client:
            mongo_client.close()
    
    print("\nTo use these photos for face recognition, restart the system:")
    print("python src/run_system.py")

if __name__ == "__main__":
    main()