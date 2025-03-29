#!/usr/bin/env python3

import os
import time
import logging
import requests
import base64
import threading
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from bson.objectid import ObjectId
import pymongo

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("photo_sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("photo_sync")

# Configuration
MONGODB_URI = "mongodb+srv://bn00017:QqqUP3%40duTjSxPu@cluster0.nh2ok3z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB = "employees"
MONGODB_COLLECTION = "patients"
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Software", "mern", "client", "public", "photos")
CHECK_INTERVAL = 10  # Seconds between database checks
LAST_SYNC_FILE = "last_sync_time.txt"

# Ensure photos directory exists
os.makedirs(PHOTOS_DIR, exist_ok=True)

def get_last_sync_time():
    """Get the timestamp of the last sync"""
    if os.path.exists(LAST_SYNC_FILE):
        try:
            with open(LAST_SYNC_FILE, "r") as f:
                timestamp_str = f.read().strip()
                return datetime.fromisoformat(timestamp_str)
        except Exception as e:
            logger.error(f"Error reading last sync time: {e}")
    
    # If file doesn't exist or there's an error, return epoch time
    return datetime.fromtimestamp(0)

def save_last_sync_time(sync_time):
    """Save the timestamp of the current sync"""
    try:
        with open(LAST_SYNC_FILE, "w") as f:
            f.write(sync_time.isoformat())
    except Exception as e:
        logger.error(f"Error saving last sync time: {e}")

def process_photo_url(patient_id, photo_url):
    """Process the photo URL and save to photos directory"""
    if not photo_url:
        logger.warning(f"No photo URL for patient {patient_id}")
        return None
    
    file_path = os.path.join(PHOTOS_DIR, f"{patient_id}.jpg")
    
    # Handle different types of photo URLs
    try:
        # Remote URL (http/https)
        if photo_url.startswith('http'):
            logger.info(f"Downloading photo for patient {patient_id} from URL")
            response = requests.get(photo_url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Saved photo to {file_path}")
                return file_path
            else:
                logger.error(f"Failed to download photo: HTTP {response.status_code}")
                return None
                
        # Base64 encoded image
        elif photo_url.startswith('data:image'):
            logger.info(f"Decoding base64 photo for patient {patient_id}")
            try:
                # Extract the base64 data
                encoded_data = photo_url.split(',')[1]
                image_data = base64.b64decode(encoded_data)
                
                # Save to file
                with open(file_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"Saved decoded photo to {file_path}")
                return file_path
            except Exception as e:
                logger.error(f"Error decoding base64 photo: {e}")
                return None
                
        # Local file path
        else:
            # Check if it's already a full path
            if os.path.exists(photo_url) and os.path.isfile(photo_url):
                logger.info(f"Copying photo from {photo_url} to {file_path}")
                with open(photo_url, 'rb') as src_file:
                    with open(file_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())
                return file_path
                
            # Check if it's a relative path in the MERN public folder
            mern_path = os.path.join(os.path.dirname(PHOTOS_DIR), photo_url)
            if os.path.exists(mern_path) and os.path.isfile(mern_path):
                logger.info(f"Copying photo from {mern_path} to {file_path}")
                with open(mern_path, 'rb') as src_file:
                    with open(file_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())
                return file_path
            
            # If it's just a filename, check if it exists in photos dir
            photos_path = os.path.join(PHOTOS_DIR, photo_url)
            if os.path.exists(photos_path) and os.path.isfile(photos_path):
                # File already exists in the right place with the right name
                if photos_path == file_path:
                    logger.info(f"Photo already exists at {file_path}")
                    return file_path
                # File exists but with wrong name
                else:
                    logger.info(f"Copying photo from {photos_path} to {file_path}")
                    with open(photos_path, 'rb') as src_file:
                        with open(file_path, 'wb') as dst_file:
                            dst_file.write(src_file.read())
                    return file_path
            
            logger.warning(f"Could not find photo file at any standard location: {photo_url}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing photo for patient {patient_id}: {e}")
        return None

def sync_photos(collection, last_sync_time):
    """Sync photos for patients created or updated since last sync"""
    current_time = datetime.utcnow()
    
    # Find patients created or updated since last sync
    # MongoDB doesn't have a default 'updated_at' field, so we need to check
    # if the collection has such fields
    try:
        # Try to find documents with timestamp fields
        query = {
            "$or": [
                {"createdAt": {"$gt": last_sync_time}},
                {"updatedAt": {"$gt": last_sync_time}}
            ]
        }
        
        patients = list(collection.find(query))
        if not patients:
            # If no patients found with timestamp fields, get all patients
            # This will happen on first run or if timestamps aren't used
            logger.info("No new patients with timestamp fields, checking all patients...")
            patients = list(collection.find({}))
    
    except Exception as e:
        logger.warning(f"Error with timestamp query: {e}")
        # Fallback to getting all patients
        patients = list(collection.find({}))
    
    logger.info(f"Found {len(patients)} patients to process")
    
    # Process each patient
    for patient in patients:
        patient_id = str(patient["_id"])
        name = patient.get("name", "Unknown")
        photo_url = patient.get("photoUrl")
        
        logger.info(f"Processing patient: {name} (ID: {patient_id})")
        
        # Process and save photo
        processed_path = process_photo_url(patient_id, photo_url)
        
        if processed_path:
            logger.info(f"Successfully processed photo for {name} (ID: {patient_id})")
        else:
            logger.warning(f"Failed to process photo for {name} (ID: {patient_id})")
    
    return current_time

def watch_collection(collection):
    """Watch MongoDB collection for changes (requires MongoDB 3.6+)"""
    try:
        # Create a change stream
        change_stream = collection.watch([
            {"$match": {"operationType": {"$in": ["insert", "update"]}}}
        ])
        
        logger.info("Watching MongoDB collection for changes")
        
        # Process each change as it occurs
        for change in change_stream:
            try:
                # Get document ID
                if change["operationType"] == "insert":
                    doc_id = str(change["fullDocument"]["_id"])
                    photo_url = change["fullDocument"].get("photoUrl")
                    name = change["fullDocument"].get("name", "Unknown")
                    logger.info(f"New patient added: {name} (ID: {doc_id})")
                else:  # update
                    doc_id = str(change["documentKey"]["_id"])
                    # Fetch updated document
                    patient = collection.find_one({"_id": ObjectId(doc_id)})
                    if patient:
                        photo_url = patient.get("photoUrl")
                        name = patient.get("name", "Unknown")
                        logger.info(f"Patient updated: {name} (ID: {doc_id})")
                    else:
                        logger.warning(f"Could not find updated patient with ID: {doc_id}")
                        continue
                
                # Process the photo
                processed_path = process_photo_url(doc_id, photo_url)
                
                if processed_path:
                    logger.info(f"Successfully processed photo for {name} (ID: {doc_id})")
                else:
                    logger.warning(f"Failed to process photo for {name} (ID: {doc_id})")
                    
            except Exception as e:
                logger.error(f"Error processing change event: {e}")
                
    except pymongo.errors.PyMongoError as e:
        logger.error(f"Error watching collection: {e}")
        logger.info("Falling back to polling mode")
        return False
        
    return True

def poll_collection(collection):
    """Poll MongoDB collection for changes periodically"""
    last_sync_time = get_last_sync_time()
    logger.info(f"Starting polling sync from {last_sync_time}")
    
    while True:
        try:
            current_time = sync_photos(collection, last_sync_time)
            save_last_sync_time(current_time)
            last_sync_time = current_time
            
            logger.info(f"Sync completed. Next sync in {CHECK_INTERVAL} seconds")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Photo sync service stopped by user")
            break
        except Exception as e:
            logger.error(f"Error during sync: {e}")
            logger.info(f"Retrying in {CHECK_INTERVAL} seconds")
            time.sleep(CHECK_INTERVAL)

def main():
    """Main function"""
    logger.info("Starting Photo Sync Service")
    logger.info("---------------------------")
    logger.info(f"MongoDB URI: {MONGODB_URI}")
    logger.info(f"Photos Directory: {PHOTOS_DIR}")
    
    # Connect to MongoDB
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]
        
        # Initial sync
        last_sync_time = get_last_sync_time()
        current_time = sync_photos(collection, last_sync_time)
        save_last_sync_time(current_time)
        
        logger.info("Initial sync completed")
        
        # Try to use change streams if available
        watch_success = False
        try:
            watch_success = watch_collection(collection)
        except Exception as e:
            logger.error(f"Error setting up change stream: {e}")
            watch_success = False
        
        # Fall back to polling if change streams not available
        if not watch_success:
            poll_collection(collection)
            
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
    
if __name__ == "__main__":
    main()