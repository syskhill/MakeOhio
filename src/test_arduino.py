#!/usr/bin/env python3
"""
Arduino Connection Test Script
This script tests connection to Arduino and allows sending commands directly.
"""

import serial
import time
import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("arduino_test")

# Arduino connection settings
BAUD_RATE = 9600
TIMEOUT = 1

def find_arduino_port():
    """Try to find Arduino on various ports"""
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
    
    # Try each port
    for port in possible_ports:
        try:
            logger.info(f"Trying port {port}...")
            arduino = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT)
            logger.info(f"SUCCESS: Connected to Arduino on {port}")
            return arduino, port
        except Exception as e:
            logger.debug(f"Could not connect to {port}: {e}")
    
    logger.error("ERROR: Could not find Arduino on any port")
    return None, None

def test_connection(arduino, port):
    """Test communication with Arduino"""
    if not arduino:
        logger.error("No Arduino connection available")
        return False
        
    logger.info("Testing communication with Arduino...")
    
    try:
        # Clear any pending data
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        
        # Send test command
        test_command = "TEST\n"
        logger.info(f"Sending: {test_command.strip()}")
        arduino.write(test_command.encode())
        
        # Wait for response
        time.sleep(1)
        
        # Read response
        response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
        logger.info(f"Response: {response}")
        
        if "TEST_RESPONSE" in response:
            logger.info("TEST SUCCESSFUL: Arduino responded as expected")
            return True
        else:
            logger.warning("TEST INCONCLUSIVE: Arduino responded but not with expected message")
            logger.warning("This might be normal if Arduino is running different firmware")
            return True
            
    except Exception as e:
        logger.error(f"ERROR testing Arduino communication: {e}")
        return False

def send_command(arduino, command):
    """Send a command to Arduino and get response"""
    if not arduino:
        logger.error("No Arduino connection available")
        return
        
    try:
        # Clear any pending data
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        
        # Send command with newline terminator
        full_command = f"{command}\n"
        logger.info(f"Sending: {command}")
        arduino.write(full_command.encode())
        
        # Wait for response
        time.sleep(0.5)
        
        # Read response
        response = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
        if response:
            logger.info(f"Response: {response}")
        else:
            logger.info("No response received")
            
    except Exception as e:
        logger.error(f"ERROR sending command: {e}")

def interactive_mode(arduino):
    """Enter interactive mode to send commands"""
    if not arduino:
        logger.error("No Arduino connection available")
        return
        
    logger.info("\n===== INTERACTIVE MODE =====")
    logger.info("Enter commands to send to Arduino. Type 'exit' to quit.")
    logger.info("Special commands:")
    logger.info("  test - Send test command")
    logger.info("  access:id,name,slot,conf - Simulate access granted")
    logger.info("  deny:id,reason,conf - Simulate access denied")
    logger.info("  message:line1,line2 - Display message on LCD")
    
    while True:
        try:
            command = input("\nCommand> ").strip()
            
            if command.lower() == 'exit':
                break
                
            if command.lower() == 'test':
                command = 'TEST'
                
            send_command(arduino, command)
            
        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Arduino Connection Test Tool")
    parser.add_argument("--port", help="Specify Arduino port")
    parser.add_argument("--command", help="Send a single command and exit")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate")
    parser.add_argument("--list", action="store_true", help="List potential ports and exit")
    args = parser.parse_args()
    
    # List potential ports
    if args.list:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            logger.info("No serial ports found")
        else:
            logger.info("Available serial ports:")
            for port in sorted(ports):
                logger.info(f"  {port.device} - {port.description}")
                
        return
    
    logger.info("Arduino Connection Test Tool")
    logger.info("==========================")
    
    # Connect to Arduino
    arduino = None
    port = None
    
    if args.port:
        # Use specified port
        try:
            logger.info(f"Trying to connect to Arduino on {args.port}...")
            arduino = serial.Serial(args.port, args.baud, timeout=TIMEOUT)
            port = args.port
            logger.info(f"SUCCESS: Connected to Arduino on {port}")
        except Exception as e:
            logger.error(f"ERROR connecting to {args.port}: {e}")
    else:
        # Auto-detect port
        arduino, port = find_arduino_port()
    
    if not arduino:
        logger.error("No Arduino connection available. Exiting.")
        return
    
    # Wait for Arduino to initialize
    logger.info("Waiting for Arduino to initialize...")
    time.sleep(2)
    
    # Test connection
    test_connection(arduino, port)
    
    # Either send single command or enter interactive mode
    if args.command:
        send_command(arduino, args.command)
    else:
        interactive_mode(arduino)
    
    # Close connection
    if arduino:
        arduino.close()
        logger.info("Arduino connection closed")

if __name__ == "__main__":
    main()