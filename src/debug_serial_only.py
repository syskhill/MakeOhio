#!/usr/bin/env python3
"""
Modified version of face_recognition_fix.py that just tests serial communication
Simplified to focus only on Arduino communication for debugging
"""

import serial
import time
import sys
import os
import logging
import threading
import argparse
import random

# ================ Configuration ================
ARDUINO_PORT = "/dev/ttyACM0"  # Default Arduino port
ARDUINO_BAUD_RATE = 9600

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("serial_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("serial_debug")

# ================ Connect to Arduino ================
arduino = None

def connect_to_arduino():
    """Try to connect to Arduino on various ports"""
    global arduino, ARDUINO_PORT
    
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
            # Save the successful port for reconnection attempts
            ARDUINO_PORT = port
            logger.info(f"* * * Connected to Arduino on {port} * * *")
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Test communication by sending a simple command
            try:
                ser.write(b"TEST\n")
                logger.info("Sent TEST command")
                time.sleep(0.5)
                response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                logger.info(f"Arduino test response: {response}")
            except Exception as e:
                logger.warning(f"Test communication error: {e}")
            
            return ser
        except Exception as e:
            logger.info(f"Could not connect to {port}: {e}")
            continue
    
    logger.warning("Could not connect to Arduino on any known port")
    return None

def send_command_to_arduino(command):
    """Send a command to the Arduino"""
    global arduino
    
    if arduino:
        try:
            # First flush any existing data
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()
            
            # Send the command with newline terminator
            arduino.write(f"{command}\n".encode())
            logger.info(f"Serial write: {command}\\n")
            
            # Wait for Arduino to process
            time.sleep(0.1)
            
            # Log the command
            logger.info(f"Sent command to Arduino: {command}")
            
            # Read any response (for debugging)
            try:
                response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
                if response.strip():
                    logger.info(f"Arduino response: {response.strip()}")
            except:
                pass
                
            return True
        except Exception as e:
            logger.error(f"Error sending command to Arduino: {e}")
            
            # Try to reconnect
            try:
                arduino.close()
            except:
                pass
                
            logger.info("Attempting to reconnect to Arduino...")
            arduino = connect_to_arduino()
            
            if arduino:
                try:
                    # Try sending the command again
                    arduino.write(f"{command}\n".encode())
                    logger.info(f"Re-sent command after reconnection: {command}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to send command after reconnection: {e}")
            
            arduino = None
            return False
    else:
        logger.warning(f"Cannot send command, Arduino not connected: {command}")
        
        # Try to establish a new connection
        logger.info("Attempting to establish new Arduino connection...")
        arduino = connect_to_arduino()
        
        if arduino:
            try:
                arduino.write(f"{command}\n".encode())
                logger.info(f"Sent command after new connection: {command}")
                return True
            except Exception as e:
                logger.error(f"Error sending command after new connection: {e}")
                arduino = None
        
        return False

def test_communication_loop():
    """Run a continuous loop testing all types of Arduino commands"""
    logger.info("Starting Arduino communication test loop")
    
    test_commands = [
        # Simple message commands
        "MESSAGE:Testing Serial,Communication",
        "MESSAGE:Hello World,Line 2",
        
        # Test commands
        "TEST",
        "STATUS",
        
        # Access commands with confidence
        "ACCESS:12345,John Smith,1,98.5",
        "ACCESS:67890,Jane Doe,2,75.3",
        
        # Deny commands with confidence
        "DENY:12345,Not pill time,45.7",
        "DENY:67890,Unknown patient,12.3"
    ]
    
    try:
        count = 0
        while True:
            # Select a command
            command = test_commands[count % len(test_commands)]
            count += 1
            
            # Add some dynamic content
            if "MESSAGE:" in command:
                timestamp = time.strftime("%H:%M:%S")
                command = f"MESSAGE:Test {count},{timestamp}"
            
            # Send the command
            logger.info(f"Test {count}: Sending {command}")
            success = send_command_to_arduino(command)
            
            if success:
                logger.info(f"Command sent successfully")
            else:
                logger.error(f"Failed to send command")
            
            # Wait between commands
            wait_time = 5
            logger.info(f"Waiting {wait_time} seconds before next command...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        logger.info("Test loop stopped by user")

def main():
    """Main function"""
    global arduino, ARDUINO_PORT, ARDUINO_BAUD_RATE
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Arduino Serial Communication Test")
    parser.add_argument("--port", help="Arduino serial port")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit")
    args = parser.parse_args()
    
    # List serial ports if requested
    if args.list:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        
        if not ports:
            print("No serial ports found")
        else:
            print("Available serial ports:")
            for port in sorted(ports):
                print(f"  {port.device} - {port.description}")
        
        return
    
    # Use specified port if provided
    if args.port:
        ARDUINO_PORT = args.port
    
    # Use specified baud rate if provided
    if args.baud:
        ARDUINO_BAUD_RATE = args.baud
    
    logger.info("Arduino Serial Communication Test")
    logger.info("===============================")
    
    # Connect to Arduino
    arduino = connect_to_arduino()
    
    if not arduino:
        logger.error("Failed to connect to Arduino. Exiting.")
        return
    
    # Run test loop
    test_communication_loop()
    
    # Clean up
    if arduino:
        arduino.close()
        logger.info("Arduino connection closed")

if __name__ == "__main__":
    main()