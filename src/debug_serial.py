#!/usr/bin/env python3
"""
Serial Communication Debug Tool
This script monitors both the face recognition system and Arduino 
to see what messages are being passed between them.
"""

import serial
import time
import sys
import os
import logging
import argparse
import threading
import subprocess
import signal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("serial_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("serial_debug")

def list_serial_ports():
    """List available serial ports"""
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    
    if not ports:
        logger.info("No serial ports found")
    else:
        logger.info("Available serial ports:")
        for port in sorted(ports):
            logger.info(f"  {port.device} - {port.description}")
    
    return [p.device for p in ports]

def monitor_arduino_serial(port, baud_rate=9600, stop_event=None):
    """Monitor Arduino serial port and log all activity"""
    try:
        arduino = serial.Serial(port, baud_rate, timeout=0.1)
        logger.info(f"Monitoring Arduino on {port} at {baud_rate} baud")
        
        # Send a test message
        try:
            arduino.write(b"TEST\n")
            logger.info(f"Sent TEST command to Arduino")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error sending TEST command: {e}")
        
        # Monitor the serial port
        while stop_event is None or not stop_event.is_set():
            try:
                if arduino.in_waiting:
                    data = arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
                    if data.strip():
                        logger.info(f"FROM ARDUINO: {data.strip()}")
                
                # Send a command to display message every 10 seconds
                current_time = time.time()
                if getattr(monitor_arduino_serial, 'last_message_time', 0) + 10 < current_time:
                    monitor_arduino_serial.last_message_time = current_time
                    message = f"Debug {int(current_time) % 100},Time: {time.strftime('%H:%M:%S')}"
                    try:
                        arduino.write(f"MESSAGE:{message}\n".encode())
                        logger.info(f"Sent periodic test message: {message}")
                    except Exception as e:
                        logger.error(f"Error sending message: {e}")
                
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error reading from Arduino: {e}")
                break
        
        arduino.close()
        logger.info("Arduino monitoring stopped")
    except Exception as e:
        logger.error(f"Error monitoring Arduino: {e}")

def monitor_python_process(face_recognition_script, stop_event=None):
    """Monitor a Python process by running in a subprocess with output captured"""
    try:
        logger.info(f"Starting face recognition script: {face_recognition_script}")
        
        # Set environment variable for debugging
        env = os.environ.copy()
        env['PILL_DISPENSER_DEBUG'] = '1'
        
        # Run the face recognition script
        process = subprocess.Popen(
            [sys.executable, face_recognition_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Log process output
        while stop_event is None or not stop_event.is_set():
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                line = line.strip()
                logger.info(f"FACE_RECOGNITION: {line}")
                
                # Look for serial communication
                if "arduino.write" in line.lower() or "sent command to arduino" in line.lower():
                    logger.info(f"SERIAL COMMUNICATION DETECTED: {line}")
        
        # Clean up
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        logger.info(f"Face recognition process exited with code {process.returncode}")
        
    except Exception as e:
        logger.error(f"Error monitoring face recognition script: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Serial Communication Debug Tool")
    parser.add_argument("--arduino-port", help="Arduino serial port")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate")
    parser.add_argument("--script", default="face_recognition_fix.py", help="Face recognition script to monitor")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    args = parser.parse_args()
    
    # List ports if requested
    if args.list_ports:
        list_serial_ports()
        return
    
    # Find Arduino port if not specified
    arduino_port = args.arduino_port
    if not arduino_port:
        ports = list_serial_ports()
        if ports:
            arduino_port = ports[0]
            logger.info(f"Selected first available port: {arduino_port}")
        else:
            logger.error("No serial ports found. Please specify --arduino-port")
            return
    
    # Find face recognition script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, args.script)
    if not os.path.exists(script_path):
        logger.error(f"Face recognition script not found: {script_path}")
        return
    
    # Set up stop event for clean shutdown
    stop_event = threading.Event()
    
    # Set up signal handler for clean shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown requested...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start threads
    arduino_thread = threading.Thread(
        target=monitor_arduino_serial, 
        args=(arduino_port, args.baud, stop_event)
    )
    
    python_thread = threading.Thread(
        target=monitor_python_process, 
        args=(script_path, stop_event)
    )
    
    arduino_thread.daemon = True
    python_thread.daemon = True
    
    arduino_thread.start()
    python_thread.start()
    
    logger.info("Monitoring started. Press Ctrl+C to stop.")
    
    try:
        # Keep main thread alive
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    finally:
        # Clean up
        stop_event.set()
        arduino_thread.join(timeout=5)
        python_thread.join(timeout=5)
        logger.info("Monitoring stopped")

if __name__ == "__main__":
    main()