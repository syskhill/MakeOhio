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
  delay(1000);  // Give the serial connection time to start
  
  // Send initial debug message
  Serial.println("Arduino starting up...");
  
  // Initialize LCD
  lcd.begin(16, 2);
  displayMessage("Starting up...", "", 2000);
  
  // Initialize other components
  Wire.begin();
  if (rtc.begin()) {
    Serial.println("RTC initialized successfully");
  } else {
    Serial.println("ERROR: Couldn't find RTC");
    displayMessage("ERROR:", "No RTC found", 2000);
  }
  
  // Test the LEDs
  Serial.println("Testing LEDs...");
  digitalWrite(GREEN_LED, HIGH);
  delay(500);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, HIGH);
  delay(500);
  digitalWrite(RED_LED, LOW);
  
  // Set up servo
  pillDispenser.attach(SERVO_PIN);
  pillDispenser.write(0); // Reset position
  Serial.println("Servo initialized");
  
  // Set up LED pins
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  
  // Send ready message
  Serial.println("SYSTEM READY - Waiting for commands");
  Serial.println("Send 'TEST' to verify communication");
  
  displayMessage("System Ready", "Camera Mode", 0);
}

void loop() {
  // Display current date and time
  displayDateTime();
  
  // Check for commands via Serial
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Echo command back for debugging
    Serial.print("Received command: ");
    Serial.println(command);
    
    if (command == "TEST") {
      // Test command to verify communication
      Serial.println("TEST_RESPONSE: Communication OK");
      digitalWrite(GREEN_LED, HIGH);
      digitalWrite(RED_LED, HIGH);
      displayMessage("Comm Test", "Success!", 2000);
      digitalWrite(GREEN_LED, LOW);
      digitalWrite(RED_LED, LOW);
    }
    else if (command.startsWith("ACCESS:")) {
      // Format: ACCESS:patient_id,patient_name,slot_number,confidence
      int commaIndex = command.indexOf(',', 7);
      if (commaIndex > 0) {
        String patientId = command.substring(7, commaIndex);
        
        // Get patient name
        String patientName = "Patient";
        int nameIndex = command.indexOf(',', commaIndex + 1);
        if (nameIndex > 0) {
          patientName = command.substring(commaIndex + 1, nameIndex);
          
          // Get slot number
          int slotIndex = command.indexOf(',', nameIndex + 1);
          int slotNumber = 1; // Default slot
          
          if (slotIndex > 0) {
            slotNumber = command.substring(nameIndex + 1, slotIndex).toInt();
            
            // Get confidence score if available
            float confidence = 0.0;
            if (slotIndex < command.length() - 1) {
              confidence = command.substring(slotIndex + 1).toFloat();
            }
            
            Serial.print("ACCESS command - PatientID: ");
            Serial.print(patientId);
            Serial.print(", Name: ");
            Serial.print(patientName);
            Serial.print(", Slot: ");
            Serial.print(slotNumber);
            Serial.print(", Confidence: ");
            Serial.println(confidence);
            
            // Grant access
            digitalWrite(GREEN_LED, HIGH);
            
            // First display confidence
            String confidenceStr = String(confidence, 1) + "%";
            displayMessage("Match: " + confidenceStr, patientName, 2000);
            
            // Then display access granted
            displayMessage("Access Granted", patientName, 2000);
            
            // Move servo to dispense from the correct slot
            rotateServoToSlot(slotNumber);
            
            digitalWrite(GREEN_LED, LOW);
            displayMessage("Pills Dispensed", "Slot: " + String(slotNumber), 3000);
          } else {
            // Old format fallback
            slotNumber = command.substring(nameIndex + 1).toInt();
            
            Serial.print("ACCESS command - PatientID: ");
            Serial.print(patientId);
            Serial.print(", Name: ");
            Serial.print(patientName);
            Serial.print(", Slot: ");
            Serial.println(slotNumber);
            
            // Grant access
            digitalWrite(GREEN_LED, HIGH);
            displayMessage("Access Granted", patientName, 2000);
            
            // Move servo to dispense from the correct slot
            rotateServoToSlot(slotNumber);
            
            digitalWrite(GREEN_LED, LOW);
            displayMessage("Pills Dispensed", "Slot: " + String(slotNumber), 3000);
          }
        } else {
          // Old format fallback
          int slotNumber = command.substring(commaIndex + 1).toInt();
          
          Serial.print("ACCESS command - PatientID: ");
          Serial.print(patientId);
          Serial.print(", Slot: ");
          Serial.println(slotNumber);
          
          // Grant access
          digitalWrite(GREEN_LED, HIGH);
          displayMessage("Access Granted", "Patient", 2000);
          
          // Move servo to dispense from the correct slot
          rotateServoToSlot(slotNumber);
          
          digitalWrite(GREEN_LED, LOW);
          displayMessage("Pills Dispensed", "Slot: " + String(slotNumber), 3000);
        }
      } else {
        Serial.println("ERROR: Invalid ACCESS command format");
      }
    } 
    else if (command.startsWith("DENY:")) {
      // Format: DENY:patient_id,reason,confidence
      String patientId = command.substring(5);
      String denyReason = "Access Denied";
      float confidence = 0.0;
      
      // Parse the components
      int firstComma = patientId.indexOf(',');
      if (firstComma > 0) {
        // Extract reason
        String remainder = patientId.substring(firstComma + 1);
        patientId = patientId.substring(0, firstComma);
        
        // Look for another comma for confidence
        int secondComma = remainder.indexOf(',');
        if (secondComma > 0) {
          // We have confidence info
          denyReason = remainder.substring(0, secondComma);
          confidence = remainder.substring(secondComma + 1).toFloat();
        } else {
          // Just reason, no confidence
          denyReason = remainder;
        }
      }
      
      Serial.print("DENY command - PatientID: ");
      Serial.print(patientId);
      Serial.print(", Reason: ");
      Serial.print(denyReason);
      if (confidence > 0) {
        Serial.print(", Confidence: ");
        Serial.println(confidence);
      } else {
        Serial.println();
      }
      
      // Deny access
      digitalWrite(RED_LED, HIGH);
      
      // Show confidence if available
      if (confidence > 0) {
        String confidenceStr = String(confidence, 1) + "%";
        displayMessage("Match: " + confidenceStr, "ID: " + patientId, 2000);
      }
      
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
  // Check if RTC is running
  if (!rtc.isrunning()) {
    static unsigned long lastRtcErrorTime = 0;
    // Only report RTC error once every 10 seconds to avoid spamming
    if (millis() - lastRtcErrorTime > 10000) {
      lastRtcErrorTime = millis();
      Serial.println("ERROR: RTC is not running!");
      displayMessage("RTC Error", "No clock found", 1000);
    }
    return;
  }
  
  DateTime now = rtc.now();
  
  // Only update display every second to avoid flicker
  static uint32_t lastDisplayUpdate = 0;
  static uint32_t lastSerialTimeUpdate = 0;
  
  // Update LCD display
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
    
    // Log time to serial once every 30 seconds for debugging
    if (millis() - lastSerialTimeUpdate >= 30000) {
      lastSerialTimeUpdate = millis();
      Serial.print("Current time: ");
      Serial.print(timeStr);
      Serial.print(" Date: ");
      Serial.println(dateStr);
    }
  }
}