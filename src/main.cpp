#include <Arduino.h>
#include <Wire.h>
#include <RTClib.h>
#include <LiquidCrystal.h>
#include <Servo.h>
#include <SPI.h>

// Pin definitions
#define GREEN_LED 4  // Green LED for access granted
#define RED_LED 5    // Red LED for access denied
#define SERVO_PIN 13 // Servo motor pin

// Initialize components
Servo pillDispenser;            // Servo for dispensing pills
RTC_DS1307 rtc;                // Real-time clock
LiquidCrystal lcd(7, 8, 9, 10, 11, 12); // LCD display

// Variables for face recognition
String recognizedPatientId = "";
bool faceRecognized = false;

// Struct to store patient data
struct Patient {
  String id;
  String name;
  int slotNumber;
};

// Array to store patients from the database
const int MAX_PATIENTS = 10;
Patient patients[MAX_PATIENTS];
int patientCount = 0;

// Function prototypes
void rotateServoToSlot(int slotNumber);
void displayMessage(String line1, String line2, int duration);
void displayDateTime();

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize LCD
  lcd.begin(16, 2);
  displayMessage("Starting up...", "", 2000);
  
  // Initialize other components
  Wire.begin();
  rtc.begin();
  
  // Set up servo
  pillDispenser.attach(SERVO_PIN);
  pillDispenser.write(0); // Reset position
  
  // Set up LED pins
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  
  displayMessage("System Ready", "Camera Mode", 0);
}

void loop() {
  // Display current date and time
  displayDateTime();
  
  // Check for commands via Serial
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("ACCESS:")) {
      // Format: ACCESS:patient_id,slot_number
      int commaIndex = command.indexOf(',', 7);
      if (commaIndex > 0) {
        String patientId = command.substring(7, commaIndex);
        int slotNumber = command.substring(commaIndex + 1).toInt();
        
        // Get patient name from MongoDB (if included in command)
        String patientName = "Patient";
        int nameIndex = command.indexOf(',', commaIndex + 1);
        if (nameIndex > 0) {
          patientName = command.substring(commaIndex + 1, nameIndex);
          slotNumber = command.substring(nameIndex + 1).toInt();
        }
        
        // Grant access
        digitalWrite(GREEN_LED, HIGH);
        displayMessage("Access Granted", patientName, 2000);
        
        // Move servo to dispense from the correct slot
        rotateServoToSlot(slotNumber);
        
        digitalWrite(GREEN_LED, LOW);
        displayMessage("Pills Dispensed", "Slot: " + String(slotNumber), 3000);
      }
    } 
    else if (command.startsWith("DENY:")) {
      // Format: DENY:patient_id or DENY:patient_id,reason
      String patientId = command.substring(5);
      String denyReason = "Access Denied";
      
      int commaIndex = patientId.indexOf(',');
      if (commaIndex > 0) {
        denyReason = patientId.substring(commaIndex + 1);
        patientId = patientId.substring(0, commaIndex);
      }
      
      // Deny access
      digitalWrite(RED_LED, HIGH);
      displayMessage("Access Denied", denyReason, 3000);
      digitalWrite(RED_LED, LOW);
    }
    else if (command == "STATUS") {
      // Send back status information to Python
      Serial.println("STATUS:READY");
    }
    else if (command.startsWith("MESSAGE:")) {
      // Custom message for LCD display
      // Format: MESSAGE:line1,line2
      String message = command.substring(8);
      int commaIndex = message.indexOf(',');
      if (commaIndex > 0) {
        String line1 = message.substring(0, commaIndex);
        String line2 = message.substring(commaIndex + 1);
        displayMessage(line1, line2, 3000);
      } else {
        displayMessage(message, "", 3000);
      }
    }
  }
  
  delay(100); // Small delay for stability
}

// Rotate servo to the correct slot
void rotateServoToSlot(int slotNumber) {
  // Map slot numbers 1-10 to servo positions 0-180
  int servoPosition = map(slotNumber, 1, 10, 0, 180);
  
  // Move to the position
  pillDispenser.write(servoPosition);
  delay(1000); // Wait for servo to reach position
  
  // Simulate pill dispensing action
  pillDispenser.write(servoPosition + 20); // Small movement to dispense
  delay(500);
  pillDispenser.write(servoPosition);
  delay(500);
  
  // Return to home position
  pillDispenser.write(0);
}

// Display message on LCD
void displayMessage(String line1, String line2, int duration) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  
  if (line2 != "") {
    lcd.setCursor(0, 1);
    lcd.print(line2);
  }
  
  if (duration > 0) {
    delay(duration);
  }
}

// Display current date and time
void displayDateTime() {
  if (!rtc.isrunning()) {
    return;
  }
  
  DateTime now = rtc.now();
  
  // Only update display every second to avoid flicker
  static uint32_t lastDisplayUpdate = 0;
  if (millis() - lastDisplayUpdate >= 1000) {
    lastDisplayUpdate = millis();
    
    String timeStr = "";
    if (now.hour() < 10) timeStr += "0";
    timeStr += String(now.hour()) + ":";
    if (now.minute() < 10) timeStr += "0";
    timeStr += String(now.minute());
    
    String dateStr = String(now.month()) + "/" + String(now.day()) + "/" + String(now.year());
    
    lcd.setCursor(0, 0);
    lcd.print("Time: " + timeStr + "    ");
    lcd.setCursor(0, 1);
    lcd.print("Date: " + dateStr);
  }
}