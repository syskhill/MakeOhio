#!/usr/bin/env python3
"""
Arduino LCD Message Test
This script sends test messages to the Arduino to verify 
that the LCD is displaying messages correctly.
"""

import serial
import time
import argparse
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_messages")

def connect_to_arduino(port, baud_rate=9600):
    """Connect to Arduino on the specified port"""
    try:
        logger.info(f"Connecting to Arduino on {port}...")
        arduino = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        logger.info("Connected to Arduino!")
        return arduino
    except Exception as e:
        logger.error(f"Failed to connect to Arduino: {e}")
        return None

def test_communication(arduino):
    """Test basic communication with Arduino"""
    if not arduino:
        logger.error("No Arduino connection")
        return False
        
    logger.info("Testing communication...")
    
    # Clear input buffer
    arduino.reset_input_buffer()
    
    # Send test command
    command = "TEST\n"
    arduino.write(command.encode())
    logger.info(f"Sent: {command.strip()}")
    
    # Wait for response
    time.sleep(1)
    
    # Read response
    if arduino.in_waiting:
        response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
        logger.info(f"Response: {response.strip()}")
        
        if "TEST_RESPONSE" in response:
            logger.info("Communication test PASSED")
            return True
        else:
            logger.warning("Unexpected response, but communication works")
            return True
    else:
        logger.warning("No response received, but serial connection is open")
        return True

def send_message_command(arduino, line1, line2="", duration=5000):
    """Send a MESSAGE command to display text on the LCD"""
    if not arduino:
        logger.error("No Arduino connection")
        return
        
    command = f"MESSAGE:{line1},{line2}\n"
    logger.info(f"Sending: {command.strip()}")
    
    # Clear buffers
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
    
    # Send command
    arduino.write(command.encode())
    
    # Wait for any response
    time.sleep(0.5)
    
    if arduino.in_waiting:
        response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
        logger.info(f"Response: {response.strip()}")
    
    # Wait for the specified duration
    logger.info(f"Message should display for {duration/1000:.1f} seconds")
    time.sleep(duration/1000)  # Convert milliseconds to seconds

def send_access_command(arduino, patient_id="12345", name="Test Patient", slot=1, confidence=95.5):
    """Send an ACCESS command to simulate face recognition"""
    if not arduino:
        logger.error("No Arduino connection")
        return
        
    command = f"ACCESS:{patient_id},{name},{slot},{confidence}\n"
    logger.info(f"Sending: {command.strip()}")
    
    # Clear buffers
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
    
    # Send command
    arduino.write(command.encode())
    
    # Wait for any response
    time.sleep(0.5)
    
    if arduino.in_waiting:
        response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
        logger.info(f"Response: {response.strip()}")
    
    # Wait for the display cycle to complete
    logger.info("Waiting for the complete access cycle (about 7 seconds)...")
    time.sleep(7)

def send_deny_command(arduino, patient_id="12345", reason="Not pill time", confidence=45.5):
    """Send a DENY command to simulate face recognition denial"""
    if not arduino:
        logger.error("No Arduino connection")
        return
        
    command = f"DENY:{patient_id},{reason},{confidence}\n"
    logger.info(f"Sending: {command.strip()}")
    
    # Clear buffers
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
    
    # Send command
    arduino.write(command.encode())
    
    # Wait for any response
    time.sleep(0.5)
    
    if arduino.in_waiting:
        response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
        logger.info(f"Response: {response.strip()}")
    
    # Wait for the display cycle to complete
    logger.info("Waiting for the complete deny cycle (about 5 seconds)...")
    time.sleep(5)

def run_message_test(arduino):
    """Run a series of test messages"""
    if not arduino:
        logger.error("No Arduino connection")
        return
        
    # Test 1: Simple message
    logger.info("\n=== Test 1: Simple Message ===")
    send_message_command(arduino, "Test Message 1", "Hello World", 3000)
    
    # Test 2: Access command
    logger.info("\n=== Test 2: Access Command ===")
    send_access_command(arduino, "12345", "John Smith", 3, 98.7)
    
    # Test 3: Deny command
    logger.info("\n=== Test 3: Deny Command ===")
    send_deny_command(arduino, "67890", "Outside pill time", 75.3)
    
    # Test 4: Long message
    logger.info("\n=== Test 4: Long Message ===")
    send_message_command(arduino, "This is a long msg", "that should scroll", 5000)
    
    logger.info("\nTest sequence complete!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Arduino LCD Message Tester")
    parser.add_argument("--port", required=True, help="Serial port for Arduino")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate")
    parser.add_argument("--message", help="Custom message to send (line1,line2)")
    args = parser.parse_args()
    
    # Connect to Arduino
    arduino = connect_to_arduino(args.port, args.baud)
    if not arduino:
        sys.exit(1)
    
    # Test communication
    if not test_communication(arduino):
        logger.error("Communication test failed")
        sys.exit(1)
    
    # Either send a custom message or run the full test
    if args.message:
        parts = args.message.split(',')
        line1 = parts[0]
        line2 = parts[1] if len(parts) > 1 else ""
        send_message_command(arduino, line1, line2, 5000)
    else:
        run_message_test(arduino)
    
    # Close the connection
    if arduino:
        arduino.close()
        logger.info("Arduino connection closed")

if __name__ == "__main__":
    main()