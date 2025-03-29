#!/usr/bin/env python3

import sys
import os
import logging
from pymongo import MongoClient
from pprint import pprint

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mongo_checker")

# MongoDB connection information
MONGODB_URI = "mongodb+srv://bn00017:QqqUP3%40duTjSxPu@cluster0.nh2ok3z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB = "employees"
MONGODB_COLLECTION = "patients"

def main():
    """Check MongoDB connection and patient records"""
    # Connect to MongoDB
    try:
        logger.info(f"Connecting to MongoDB: {MONGODB_URI}")
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]
        logger.info("Connected to MongoDB successfully!")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return
    
    # Check patient records
    try:
        # Count patients
        patient_count = collection.count_documents({})
        logger.info(f"Found {patient_count} patients in database")
        
        # Get all patients
        patients = list(collection.find({}))
        logger.info("\n==== Patient Information ====")
        
        for i, patient in enumerate(patients):
            print(f"\nPatient {i+1}:")
            print(f"  ID: {patient['_id']}")
            print(f"  Name: {patient.get('name', 'Unknown')}")
            print(f"  Slot Number: {patient.get('slotNumber', 'Not specified')}")
            
            # Check photo URL
            photo_url = patient.get('photoUrl', None)
            if photo_url:
                print(f"  Photo URL: {photo_url}")
                
                # Analyze URL type
                if photo_url.startswith('http'):
                    print(f"  URL Type: Remote URL (will be downloaded)")
                elif photo_url.startswith('data:image'):
                    print(f"  URL Type: Base64 encoded image")
                else:
                    print(f"  URL Type: Local file path")
                    
                    # Check if file exists in common locations
                    found = False
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    
                    possible_paths = [
                        photo_url,  # Absolute path
                        os.path.join(base_path, photo_url),  # Relative to script
                        os.path.join(base_path, "..", "Software", "mern", "client", "public", photo_url),  # Relative to MERN public
                        os.path.join(base_path, "..", "Software", "mern", "client", "public", "photos", photo_url),  # In photos dir
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            print(f"  Found at: {path}")
                            found = True
                            break
                    
                    if not found:
                        print(f"  WARNING: Could not locate file at any standard location")
            else:
                print(f"  Photo URL: Not specified (REQUIRED for face recognition)")
    
    except Exception as e:
        logger.error(f"Error accessing patient records: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
