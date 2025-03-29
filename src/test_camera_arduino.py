import cv2
import serial
import time
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_camera_arduino")

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

def test_camera():
    """Test if the camera can be accessed"""
    logger.info("Testing camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return False
        
        logger.info("Camera opened successfully")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read frame from camera")
            cap.release()
            return False
        
        logger.info("Successfully read frame from camera")
        
        # Save a test image
        cv2.imwrite("test_camera.jpg", frame)
        logger.info("Saved test image to test_camera.jpg")
        
        # Display image (will be visible if running in desktop environment)
        cv2.imshow('Test Camera', frame)
        cv2.waitKey(3000)  # Show for 3 seconds
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    except Exception as e:
        logger.error(f"Error testing camera: {e}")
        return False

def test_arduino_communication(arduino):
    """Test communication with Arduino"""
    if not arduino:
        logger.error("Arduino not connected")
        return False
    
    try:
        # First, flush any existing data
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        
        # Send test command
        test_command = "TEST\n"
        logger.info(f"Sending command to Arduino: {test_command.strip()}")
        arduino.write(test_command.encode())
        
        # Wait for response
        time.sleep(1)
        
        # Read response
        response = arduino.read(arduino.in_waiting).decode('utf-8')
        logger.info(f"Response from Arduino: {response}")
        
        # Check if we got a valid response
        if "TEST_RESPONSE" in response:
            logger.info("Arduino communication test PASSED")
            return True
        else:
            logger.error("Arduino did not respond as expected")
            return False
    
    except Exception as e:
        logger.error(f"Error communicating with Arduino: {e}")
        return False

def test_lcd_message(arduino):
    """Test sending message to LCD"""
    if not arduino:
        logger.error("Arduino not connected")
        return False
    
    try:
        message_command = "MESSAGE:Camera Test,Running...\n"
        logger.info(f"Sending LCD message: {message_command.strip()}")
        arduino.write(message_command.encode())
        time.sleep(2)
        return True
    except Exception as e:
        logger.error(f"Error sending LCD message: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Camera and Arduino Test")
    logger.info("---------------------------------")
    
    # Test camera
    camera_result = test_camera()
    if not camera_result:
        logger.error("Camera test FAILED")
    else:
        logger.info("Camera test PASSED")
    
    # Connect to Arduino
    arduino = connect_to_arduino()
    
    # Test Arduino communication
    if arduino:
        comm_result = test_arduino_communication(arduino)
        if comm_result:
            # Try sending a message to LCD
            lcd_result = test_lcd_message(arduino)
            if lcd_result:
                logger.info("LCD message test PASSED")
            else:
                logger.error("LCD message test FAILED")
        
        # Close the connection
        arduino.close()
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()