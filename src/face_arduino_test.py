#!/usr/bin/env python3
"""
Simplified Face Recognition + Arduino Test
This script tests the camera, face detection, and Arduino communication together.
"""

import cv2
import numpy as np
import serial
import time
import os
import sys
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("face_arduino.log")
    ]
)
logger = logging.getLogger("face_arduino")

# ================ Configuration ================
CAMERA_ID = 0
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
ARDUINO_BAUD_RATE = 9600

# ================ Arduino Connection ================
def connect_to_arduino():
    """Try to connect to Arduino on various ports"""
    possible_ports = [
        # Linux ports
        "/dev/ttyACM0", "/dev/ttyACM1", 
        "/dev/ttyUSB0", "/dev/ttyUSB1",
        # Windows ports
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8"
    ]
    
    # Add MacOS-specific ports
    if sys.platform == 'darwin':
        for i in range(10):
            possible_ports.append(f"/dev/tty.usbmodem{i+1}")
            possible_ports.append(f"/dev/tty.usbserial{i+1}")
    
    logger.info(f"Searching for Arduino on ports: {possible_ports}")
    
    for port in possible_ports:
        try:
            logger.info(f"Trying to connect to Arduino on {port}...")
            ser = serial.Serial(port, ARDUINO_BAUD_RATE, timeout=1)
            logger.info(f"* * * Connected to Arduino on {port} * * *")
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Test communication by sending a simple command
            try:
                ser.write(b"MESSAGE:Arduino Test,Connected!\n")
                logger.info("Sent test message to Arduino")
                time.sleep(0.5)
                response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                if response:
                    logger.info(f"Arduino response: {response}")
            except Exception as e:
                logger.warning(f"Test communication error: {e}")
            
            return ser, port
        except Exception as e:
            logger.info(f"Could not connect to {port}: {e}")
            continue
    
    logger.warning("Could not connect to Arduino on any known port")
    return None, None

def send_command_to_arduino(arduino, command):
    """Send a command to the Arduino"""
    if not arduino:
        logger.warning(f"Cannot send command, Arduino not connected: {command}")
        return False
    
    try:
        # First flush any existing data
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        
        # Send the command with newline terminator
        full_command = f"{command}\n"
        arduino.write(full_command.encode())
        
        # Log the command
        logger.info(f"Sent command to Arduino: {command}")
        
        # Read any response (for debugging)
        time.sleep(0.1)
        if arduino.in_waiting:
            response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
            if response.strip():
                logger.info(f"Arduino response: {response.strip()}")
        
        return True
    except Exception as e:
        logger.error(f"Error sending command to Arduino: {e}")
        return False

# ================ Face Detection ================
def setup_face_detection():
    """Initialize face detection"""
    try:
        # Load the face cascade classifier
        cascade_path = cv2.data.haarcascades + FACE_CASCADE_PATH
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error(f"Failed to load face cascade classifier")
            return None
        
        logger.info("Face detection initialized successfully")
        return face_cascade
    except Exception as e:
        logger.error(f"Error setting up face detection: {e}")
        return None

def setup_camera(camera_id=0):
    """Initialize camera with optimized settings"""
    try:
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return None
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Small buffer for reduced latency
        
        # Log actual camera settings
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera opened with settings: {actual_width}x{actual_height}, {actual_fps} FPS")
        return cap
    except Exception as e:
        logger.error(f"Error setting up camera: {e}")
        return None

def run_face_detection(camera, face_cascade, arduino, arduino_port):
    """Run face detection and send Arduino commands when faces are detected"""
    if not camera or not face_cascade:
        logger.error("Missing camera or face cascade")
        return
    
    logger.info("Starting face detection loop")
    face_detected = False
    last_command_time = 0
    command_cooldown = 3.0  # Seconds between commands
    frame_count = 0
    fps_time = time.time()
    
    try:
        while True:
            # Read frame from camera
            ret, frame = camera.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                # Try to reconnect
                camera.release()
                time.sleep(1)
                camera = setup_camera(CAMERA_ID)
                if not camera:
                    logger.error("Failed to reconnect to camera")
                    break
                continue
            
            # Count frames for FPS calculation
            frame_count += 1
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                fps = frame_count / (current_time - fps_time)
                logger.info(f"Camera running at {fps:.1f} FPS")
                frame_count = 0
                fps_time = current_time
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process faces and send commands to Arduino
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Only send commands every few seconds to avoid flooding
                if current_time - last_command_time > command_cooldown:
                    last_command_time = current_time
                    
                    # Alternate between access and deny for testing
                    if face_detected:
                        # Send access command
                        face_detected = False
                        confidence = 85.5  # Random confidence value for testing
                        command = f"ACCESS:12345,Face Detected,1,{confidence}"
                        logger.info(f"Face detected - sending ACCESS command")
                        success = send_command_to_arduino(arduino, command)
                        
                        if not success and arduino:
                            # Try to reconnect to Arduino
                            try:
                                arduino.close()
                            except:
                                pass
                            
                            logger.info("Reconnecting to Arduino...")
                            arduino, arduino_port = connect_to_arduino()
                    else:
                        # Send deny command
                        face_detected = True
                        confidence = 45.5  # Random confidence value for testing
                        command = f"DENY:12345,Test Deny,{confidence}"
                        logger.info(f"Face detected - sending DENY command")
                        success = send_command_to_arduino(arduino, command)
                        
                        if not success and arduino:
                            # Try to reconnect to Arduino
                            try:
                                arduino.close()
                            except:
                                pass
                            
                            logger.info("Reconnecting to Arduino...")
                            arduino, arduino_port = connect_to_arduino()
            
            # Display the number of faces detected
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Display frame
            cv2.imshow("Face Detection Test", frame)
            
            # Check for key press to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            
            # Send a message to Arduino every 10 seconds regardless of face detection
            if int(current_time) % 10 == 0 and abs(current_time - int(current_time)) < 0.1:
                message = f"Heartbeat,Time: {time.strftime('%H:%M:%S')}"
                send_command_to_arduino(arduino, f"MESSAGE:{message}")
    
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Error in face detection loop: {e}")
    finally:
        # Clean up
        if camera:
            camera.release()
        
        cv2.destroyAllWindows()
        
        if arduino:
            send_command_to_arduino(arduino, "MESSAGE:Test Complete,Goodbye!")
            time.sleep(1)
            arduino.close()
            
        logger.info("Face detection test completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Face Detection + Arduino Test")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID to use")
    parser.add_argument("--port", help="Arduino port to use (auto-detect if not specified)")
    parser.add_argument("--baud", type=int, default=9600, help="Arduino baud rate")
    args = parser.parse_args()
    
    logger.info("Face Detection + Arduino Test")
    logger.info("============================")
    
    # Update configuration from arguments
    global CAMERA_ID, ARDUINO_BAUD_RATE
    CAMERA_ID = args.camera
    ARDUINO_BAUD_RATE = args.baud
    
    # Setup components
    face_cascade = setup_face_detection()
    if not face_cascade:
        logger.error("Failed to set up face detection. Exiting.")
        return
    
    # Either use specified port or auto-detect
    arduino = None
    arduino_port = None
    
    if args.port:
        try:
            arduino = serial.Serial(args.port, ARDUINO_BAUD_RATE, timeout=1)
            arduino_port = args.port
            logger.info(f"Connected to Arduino on specified port: {args.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Arduino on port {args.port}: {e}")
    else:
        arduino, arduino_port = connect_to_arduino()
    
    # Send test message if Arduino connected
    if arduino:
        send_command_to_arduino(arduino, "MESSAGE:Test Starting,Face Detection")
    else:
        logger.warning("Running without Arduino connection")
    
    # Setup camera
    camera = setup_camera(CAMERA_ID)
    if not camera:
        logger.error("Failed to set up camera. Exiting.")
        
        # Clean up
        if arduino:
            arduino.close()
            
        return
    
    # Run face detection loop
    run_face_detection(camera, face_cascade, arduino, arduino_port)

if __name__ == "__main__":
    main()