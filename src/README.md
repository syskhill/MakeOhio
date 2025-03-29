# Face Recognition System

This is a face recognition system for a medication pill dispenser. It recognizes patients and dispenses medication based on scheduled pill times.

## New Two-Step Usage Process

The face recognition system now uses a two-step process:

1. **Training**: Run `train_face_model.py` to load patient photos, train the model, and save it
2. **Recognition**: Run `run_face_recognition.py` to use the pre-trained model for face recognition

This separation allows the recognition to run efficiently without requiring retraining each time.

## Training the Model

Train the model using patient photos:

```bash
python src/train_face_model.py
```

Options:
- `--photos-dir PATH`: Path to directory containing patient photos
- `--model PATH`: Path to save the trained model
- `--mapping PATH`: Path to save the patient mapping
- `--no-mongodb`: Skip MongoDB and load patients from photo directory
- `--verify`: Verify the model accuracy after training
- `--test-dir PATH`: Directory containing test photos

## Running Face Recognition

Run the face recognition system using the pre-trained model:

```bash
python src/run_face_recognition.py
```

Options:
- `--camera ID`: Camera ID to use (default: 0)
- `--threshold VALUE`: Recognition confidence threshold (default: 20)
- `--model PATH`: Path to trained model file
- `--mapping PATH`: Path to patient mapping file
- `--arduino-port PORT`: Arduino serial port
- `--no-arduino`: Run without Arduino connection
- `--debug`: Enable debug mode

## Diagnostic Tools

Several diagnostic tools are available:

- `test_camera.py`: Test camera access and performance
- `test_arduino.py`: Test Arduino connectivity
- `face_arduino_test.py`: Test camera and Arduino together
- `debug_serial.py`: Monitor serial communication
- `test_messages.py`: Test LCD messages on Arduino

## Photo Requirements

For best results:
- Place patient photos in the photos directory
- Use filenames in one of these formats:
  - `patient_id.jpg` (single photo)
  - `patient_id_1.jpg`, `patient_id_2.jpg`, etc. (multiple photos)
- Ensure photos have good lighting and clear face visibility
- Multiple photos per patient will improve recognition accuracy

## Troubleshooting

If face recognition doesn't work:
1. Check if the model was trained successfully
2. Verify camera connectivity with `test_camera.py`
3. Check Arduino connection with `test_arduino.py`
4. Test messages on the LCD with `test_messages.py`
5. Try running in debug mode: `python src/run_face_recognition.py --debug`