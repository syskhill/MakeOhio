# Pill Dispenser System with Face Recognition and Time-Based Access

This system combines a MERN (MongoDB, Express, React, Node.js) stack web application with an Arduino-based pill dispenser that uses face recognition and time-based access control.

## System Components

1. **Web Application**: MERN stack for patient management
   - Add/edit/delete patient records
   - Upload patient photos
   - Set pill schedules and slot numbers
   - Admin interface for viewing all data

2. **Arduino Hardware**:
   - Real-time clock for accurate time tracking
   - Servo motor for dispensing pills from different slots
   - LCD display for status information
   - RFID reader for testing/backup authentication
   - LEDs for visual feedback

3. **Face Recognition System**:
   - Python script using OpenCV
   - Communicates with both the MongoDB database and Arduino
   - Enforces time-based access control

## Setup Instructions

### Web Application (MERN Stack)

1. **Install Dependencies**:
   ```bash
   # Server
   cd mern/server
   npm install

   # Client
   cd ../client
   npm install
   ```

2. **Start the Server**:
   ```bash
   cd ../server
   npm start
   ```

3. **Start the Client**:
   ```bash
   cd ../client
   npm run dev
   ```

4. **Access the Web Interface**:
   - Patient Management: http://localhost:5173/
   - Add New Patient: http://localhost:5173/form
   - Admin Dashboard: http://localhost:5173/admin

### Arduino Setup

1. **Hardware Requirements**:
   - Arduino board (with Wi-Fi capability, e.g., ESP8266 or ESP32)
   - Servo motor for dispensing
   - LCD display (16x2)
   - Real-time clock module (DS1307)
   - RGB LEDs for status indication
   - Camera module for face recognition

2. **Install Required Libraries**:
   - Via Arduino IDE or PlatformIO:
     - RTClib
     - LiquidCrystal
     - Servo
     - SPI
     - MFRC522 (for RFID)
     - ESP8266WiFi (or equivalent for your board)
     - ArduinoJson

3. **Upload the Code**:
   - Open `Hardware/src/main_with_api.cpp` in Arduino IDE or PlatformIO
   - Update WiFi credentials and server IP address
   - Upload to your Arduino board

### Face Recognition Setup

1. **Install Required Python Libraries**:
   ```bash
   pip install opencv-python numpy requests pyserial
   ```

2. **Run the Face Recognition System**:
   ```bash
   cd Hardware/src
   python face_recognition.py
   ```

## Usage

### Adding a Patient

1. Navigate to http://localhost:5173/form
2. Fill in patient details:
   - Name
   - Upload a clear photo of their face
   - Enter pill times in comma-separated format (e.g., "8:00,12:00,18:00")
   - Enter slot number where their pills will be stored

### Time-Based Access Control

The system enforces time-based access control:
- Patients can only access their pills within ±1 hour of their scheduled times
- The Arduino checks the current time against scheduled pill times
- Access is only granted when a face is recognized AND the time is appropriate

### Admin Interface

The admin dashboard provides:
- A comprehensive view of all patient data
- Access to edit or delete records
- System status information

## API Endpoints

### Patient Data

- `GET /record/arduino/patients` - Returns list of all patients with their details
- `GET /record/arduino/access/:id` - Checks if a patient has access at the current time

## Troubleshooting

### Web Application
- Check that MongoDB is running
- Verify API endpoints using tools like Postman
- Check browser console for JavaScript errors

### Arduino
- Check serial monitor for debugging information
- Verify WiFi connection is established
- Ensure correct server IP address is configured

### Face Recognition
- Ensure good lighting for face detection
- Use high-quality photos for training
- Adjust confidence threshold as needed

## Security Considerations

- This system handles patient medical information and should be secured appropriately
- Consider implementing HTTPS for the web application
- Password-protect the admin interface in production
- Encrypt sensitive data in the database