#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_system")

def run_command(command, background=False):
    """Run a command and return the process"""
    logger.info(f"Running: {command}")
    
    if background:
        # Run in background
        if sys.platform == 'win32':
            process = subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            process = subprocess.Popen(command, shell=True)
        return process
    else:
        # Run in foreground
        try:
            subprocess.run(command, shell=True, check=True)
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with code {e.returncode}")
            return None

def main():
    """Main function to run the face recognition system"""
    parser = argparse.ArgumentParser(description="Run the pill dispenser face recognition system")
    parser.add_argument("--sync-only", action="store_true", help="Only sync photos, don't start face recognition")
    parser.add_argument("--check-only", action="store_true", help="Only check MongoDB, don't start anything")
    parser.add_argument("--no-sync", action="store_true", help="Don't start photo sync service")
    parser.add_argument("--test", action="store_true", help="Run test recognition instead of full system")
    args = parser.parse_args()
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Check MongoDB configuration
    logger.info("[Step 1/4] Checking MongoDB patient data...")
    check_cmd = f"python {os.path.join(script_dir, 'check_mongo.py')}"
    run_command(check_cmd)
    
    if args.check_only:
        logger.info("Check only mode - exiting after MongoDB check")
        return
    
    # Step 2: Start photo sync service (unless disabled)
    sync_process = None
    if not args.no_sync:
        logger.info("[Step 2/4] Starting photo sync service...")
        sync_cmd = f"python {os.path.join(script_dir, 'photo_sync.py')}"
        sync_process = run_command(sync_cmd, background=True)
        
        # Give the sync service time to process photos
        logger.info("Waiting 5 seconds for initial photo sync...")
        time.sleep(5)
    else:
        logger.info("[Step 2/4] Photo sync service disabled")
    
    if args.sync_only:
        logger.info("Sync only mode - system will continue running the sync service in the background")
        logger.info("Press Ctrl+C in the sync window to stop the service")
        return
    
    # Step 3: Run face recognition
    if args.test:
        logger.info("[Step 3/4] Running face recognition test...")
        rec_cmd = f"python {os.path.join(script_dir, 'test_recognition.py')}"
    else:
        logger.info("[Step 3/4] Starting face recognition system...")
        rec_cmd = f"python {os.path.join(script_dir, 'face_recognition_fix.py')}"
    
    rec_process = run_command(rec_cmd, background=True)
    
    # Step 4: Monitor processes
    logger.info("[Step 4/4] System running - press Ctrl+C to stop")
    try:
        while True:
            # Check if processes are still running
            if sync_process and sync_process.poll() is not None:
                logger.warning("Photo sync service has stopped")
            
            if rec_process and rec_process.poll() is not None:
                logger.warning("Face recognition has stopped")
                break
            
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
        
        # Terminate processes
        if sync_process:
            sync_process.terminate()
        if rec_process:
            rec_process.terminate()
            
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
