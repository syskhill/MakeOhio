#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import logging
import argparse
import atexit

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
            # On Windows, use CREATE_NEW_CONSOLE to create a new window
            process = subprocess.Popen(
                command, 
                shell=True, 
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # On Linux/Mac, need to set up env vars properly for shell execution
            # and avoid breaking the environment
            process = subprocess.Popen(
                command,
                shell=True,
                env=os.environ.copy(),  # Use current environment
                stdout=None,  # Keep stdout/stderr connected to parent
                stderr=None
            )
        
        # Give process time to start
        time.sleep(1)
        
        # Check if process started successfully
        if process.poll() is not None:
            logger.error(f"Process failed to start or terminated immediately with code {process.returncode}")
        else:
            logger.info(f"Process started with PID {process.pid}")
            
        return process
    else:
        # Run in foreground
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                env=os.environ.copy()
            )
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
    parser.add_argument("--debug", action="store_true", help="Run with debug mode enabled")
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
        
        # Create the command with appropriate syntax for the platform
        if sys.platform == 'win32':
            sync_cmd = f"start cmd /k python {os.path.join(script_dir, 'photo_sync.py')}"
        else:
            # For Linux/Mac
            terminal_app = "gnome-terminal"
            # Check if we're on macOS
            if sys.platform == 'darwin':
                terminal_app = "open -a Terminal"
            
            # Try to detect terminal application
            for term in ["gnome-terminal", "xterm", "konsole", "terminator"]:
                try:
                    subprocess.run(["which", term], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    terminal_app = term
                    break
                except:
                    continue
                    
            # Create command for detected terminal
            sync_cmd = f"{terminal_app} -- python {os.path.join(script_dir, 'photo_sync.py')}"
            
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
    
    # Step 3: Run face recognition in a separate window
    if args.test:
        logger.info("[Step 3/4] Running face recognition test...")
        script_name = 'test_recognition.py'
    elif args.debug:
        logger.info("[Step 3/4] Starting face recognition system in DEBUG mode...")
        script_name = 'debug_recognition.py'
    else:
        logger.info("[Step 3/4] Starting face recognition system with debug overlay...")
        script_name = 'face_recognition_fix.py'
        
    # Create a platform-specific command to launch in a new window
    if sys.platform == 'win32':
        # Windows: use start cmd to create new console window
        if script_name == 'face_recognition_fix.py':
            # Debug mode with environment variable
            rec_cmd = f"start cmd /k set PILL_DISPENSER_DEBUG=1 && python {os.path.join(script_dir, script_name)}"
        else:
            rec_cmd = f"start cmd /k python {os.path.join(script_dir, script_name)}"
    else:
        # Linux/Mac: find available terminal
        terminal_app = "gnome-terminal"
        
        # Check if we're on macOS
        if sys.platform == 'darwin':
            terminal_app = "open -a Terminal"
        
        # Try to detect terminal application
        for term in ["gnome-terminal", "xterm", "konsole", "terminator"]:
            try:
                subprocess.run(["which", term], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                terminal_app = term
                break
            except:
                continue
                
        # Create command with appropriate shell syntax for setting env var
        if script_name == 'face_recognition_fix.py':
            # Debug mode with environment variable
            rec_cmd = f"{terminal_app} -- bash -c \"PILL_DISPENSER_DEBUG=1 python {os.path.join(script_dir, script_name)}\""
        else:
            rec_cmd = f"{terminal_app} -- python {os.path.join(script_dir, script_name)}"
    
    rec_process = run_command(rec_cmd, background=True)
    
    # Register cleanup function to ensure processes are terminated
    def cleanup_processes():
        logger.info("Cleaning up processes...")
        if sync_process:
            try:
                sync_process.terminate()
                logger.info("Terminated photo sync service")
            except:
                pass
        if rec_process:
            try:
                rec_process.terminate()
                logger.info("Terminated face recognition")
            except:
                pass
        logger.info("Cleanup complete")
    
    atexit.register(cleanup_processes)
    
    # Log the state of the launched processes
    if rec_process:
        logger.info(f"Face recognition running with PID: {rec_process.pid}")
    if sync_process:
        logger.info(f"Photo sync service running with PID: {sync_process.pid}")
    
    # Step 4: Monitor processes
    logger.info("[Step 4/4] System running - press Ctrl+C to stop")
    logger.info("Note: Each component is running in its own window")
    
    try:
        # Keep main process running
        print("\nSystem is now running in multiple windows.")
        print("Press Ctrl+C here to stop all components.\n")
        
        while True:
            # Print status update every 10 seconds
            for i in range(10):
                # Check if processes are still running
                if sync_process and sync_process.poll() is not None:
                    logger.warning("Photo sync service has stopped")
                
                if rec_process and rec_process.poll() is not None:
                    logger.warning("Face recognition has stopped")
                    print("Face recognition window closed. Restarting...")
                    # Restart face recognition in a new window
                    if args.test:
                        script_name = 'test_recognition.py'
                    elif args.debug:
                        script_name = 'debug_recognition.py'
                    else:
                        script_name = 'face_recognition_fix.py'
                        
                    # Create platform-specific restart command for new window
                    if sys.platform == 'win32':
                        if script_name == 'face_recognition_fix.py':
                            rec_cmd = f"start cmd /k set PILL_DISPENSER_DEBUG=1 && python {os.path.join(script_dir, script_name)}"
                        else:
                            rec_cmd = f"start cmd /k python {os.path.join(script_dir, script_name)}"
                    else:
                        # Find terminal application again
                        terminal_app = "gnome-terminal"
                        for term in ["gnome-terminal", "xterm", "konsole", "terminator"]:
                            try:
                                subprocess.run(["which", term], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                                terminal_app = term
                                break
                            except:
                                continue
                                
                        if script_name == 'face_recognition_fix.py':
                            rec_cmd = f"{terminal_app} -- bash -c \"PILL_DISPENSER_DEBUG=1 python {os.path.join(script_dir, script_name)}\""
                        else:
                            rec_cmd = f"{terminal_app} -- python {os.path.join(script_dir, script_name)}"
                    rec_process = run_command(rec_cmd, background=True)
                
                time.sleep(1)
            
            # Print status every 10 seconds
            print(f"System running... (Press Ctrl+C to stop)")
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        logger.info("Shutting down system...")
        cleanup_processes()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
