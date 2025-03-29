import cv2
import time
import os
import logging
import serial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_face_detection")

# Arduino connection settings
ARDUINO_PORT = "COM4"  # Use the correct COM port
ARDUINO_BAUD_RATE = 9600

def connect_to_arduino():
    """Try to connect to Arduino"""
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
        logger.info(f"Connected to Arduino on {ARDUINO_PORT}")
        time.sleep(2)  # Wait for Arduino to initialize
        return arduino
    except Exception as e:
        logger.error(f"Error connecting to Arduino: {e}")
        return None

def run_face_detection():
    """Run basic face detection with the camera"""
    try:
        # Connect to Arduino
        arduino = connect_to_arduino()
        if arduino:
            # Send message to Arduino
            arduino.write(b"MESSAGE:Face Detection,Starting...\n")
        
        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("Loaded face detection model")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        logger.info("Camera opened successfully")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        face_detected = False
        start_time = time.time()
        
        while time.time() - start_time < 60:  # Run for 60 seconds
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame from camera")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Draw rectangle around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_detected = True
                
                # Send message to Arduino
                if arduino:
                    arduino.write(b"MESSAGE:Face Detected,Processing...\n")
                
                logger.info(f"Face detected at position: x={x}, y={y}, width={w}, height={h}")
            
            # Display the frame
            cv2.imshow('Face Detection Test', frame)
            
            # Save a frame with faces every 5 seconds
            if face_detected and int(time.time()) % 5 == 0:
                filename = f"face_detected_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Saved face detection image to {filename}")
                face_detected = False
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        if arduino:
            arduino.write(b"MESSAGE:Face Detection,Completed\n")
            arduino.close()
        
        logger.info("Face detection test completed")
    
    except Exception as e:
        logger.error(f"Error in face detection: {e}")

if __name__ == "__main__":
    logger.info("Starting Face Detection Test")
    logger.info("---------------------------")
    run_face_detection()