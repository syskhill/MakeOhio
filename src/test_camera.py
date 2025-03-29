#!/usr/bin/env python3
"""
Camera Test Tool
This script tests the camera without face recognition to diagnose camera issues.
"""

import cv2
import time
import sys
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("camera_test.log")
    ]
)
logger = logging.getLogger("camera_test")

def test_camera(camera_id=0, resolution=(640, 480), fps=30):
    """Test camera access with specified parameters"""
    logger.info(f"Testing camera ID {camera_id} at {resolution[0]}x{resolution[1]}, {fps} FPS")
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return False
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Set buffer size to 1 to reduce lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual settings
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera opened with actual settings: {actual_width}x{actual_height}, {actual_fps} FPS")
        
        # Test reading frames
        frame_count = 0
        start_time = time.time()
        fps_time = start_time
        
        logger.info("Starting to capture frames...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"Failed to read frame {frame_count+1}")
                # Try to reconnect to camera if read fails
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    logger.error("Failed to reconnect to camera")
                    return False
                continue
            
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Calculate and log FPS every second
            if current_time - fps_time >= 1.0:
                fps_rate = frame_count / elapsed
                logger.info(f"Captured {frame_count} frames in {elapsed:.1f} seconds ({fps_rate:.1f} FPS)")
                fps_time = current_time
            
            # Add text to frame
            cv2.putText(
                frame,
                f"Frame: {frame_count} | FPS: {frame_count/elapsed:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display frame
            cv2.imshow('Camera Test', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            
            # Save a test image every 30 frames
            if frame_count % 30 == 0:
                filename = f"test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Saved test frame to {filename}")
            
            # Cap test at 300 frames (about 10 seconds at 30 FPS)
            if frame_count >= 300:
                logger.info("Reached frame limit")
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Camera test completed successfully. Captured {frame_count} frames.")
        return True
        
    except Exception as e:
        logger.error(f"Error testing camera: {e}")
        try:
            cap.release()
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
            
        return False

def list_cameras():
    """Attempt to list available cameras"""
    logger.info("Searching for available cameras...")
    
    max_test = 5  # Test up to this many camera indices
    available_cameras = []
    
    for i in range(max_test):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                logger.info(f"Camera index {i} is available")
                available_cameras.append(i)
                cap.release()
            else:
                logger.info(f"Camera index {i} is not available")
        except Exception as e:
            logger.error(f"Error checking camera {i}: {e}")
    
    return available_cameras

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Camera Test Tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID to test")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--list", action="store_true", help="List available cameras and exit")
    args = parser.parse_args()
    
    logger.info("Camera Test Tool Starting")
    logger.info("=======================")
    logger.info(f"OpenCV version: {cv2.__version__}")
    
    # List cameras if requested
    if args.list:
        available_cameras = list_cameras()
        if available_cameras:
            logger.info(f"Found {len(available_cameras)} available cameras: {available_cameras}")
        else:
            logger.warning("No cameras found")
        return
    
    # Test the specified camera
    test_camera(
        camera_id=args.camera,
        resolution=(args.width, args.height),
        fps=args.fps
    )

if __name__ == "__main__":
    main()