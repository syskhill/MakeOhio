# Face Recognition Photo Management

This directory contains scripts for face recognition and photo management.

## Files

- `face_recognition.py` - Main face recognition system
- `face_recognition_fix.py` - Version with dependency checking and error handling
- `check_mongo.py` - Script to check MongoDB patient data and photo URLs
- `photo_sync.py` - Background service to automatically sync photos from MongoDB
- `test_recognition.py` - Diagnostic tool for testing face recognition performance

## Requirements

Install required dependencies:

```bash
pip install opencv-contrib-python pymongo pyserial flask requests
```

## Photo Management

Photos for face recognition should be stored in:
```
/Software/mern/client/public/photos/
```

Each patient record in MongoDB should have a `photoUrl` field containing one of:
1. A local file name (e.g., "john.jpg")
2. A full URL (e.g., "https://example.com/photos/john.jpg")
3. A base64 encoded image (e.g., "data:image/jpeg;base64,...")

## Tools

### Check MongoDB Patient Data
Inspect MongoDB patient records and photo URLs:

```bash
python src/check_mongo.py
```

### Automatic Photo Sync
Run this in the background to automatically download/sync photos when new patients are added:

```bash
python src/photo_sync.py
```

This service will:
- Monitor MongoDB for new or updated patient records
- Download photos from URLs or decode base64 images
- Save all photos to the correct folder with proper naming
- Run continuously to keep photos in sync

### Test Face Recognition
Test face recognition performance and analyze results:

```bash
python src/test_recognition.py
```

### Main Face Recognition System
Run the main face recognition system:

```bash
python src/face_recognition_fix.py
```

This will connect to:
- MongoDB for patient data
- Arduino for dispenser control
- Camera for face recognition