#include <Arduino.h>
#include <Wire.h>
#include <RTClib.h>
#include <LiquidCrystal.h>
#include <Servo.h>
#include <SPI.h>
#include <MFRC522.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>

// Wi-Fi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// API endpoint - replace with your server IP address
const char* apiBaseUrl = "http://YOUR_SERVER_IP:5050/record/arduino";

// Pin definitions
#define SS_PIN 53   // RC522 SDA (Slave Select)
#define RST_PIN 3   // RC522 Reset
#define GREEN_LED 4 // Green LED for access granted
#define RED_LED 5   // Red LED for access denied
#define SERVO_PIN 13 // Servo motor pin
#define CAMERA_RX 2  // Camera module RX pin
#define CAMERA_TX 1  // Camera module TX pin

// Initialize components
MFRC522 rfid(SS_PIN, RST_PIN); // RFID reader
Servo pillDispenser;            // Servo for dispensing pills
RTC_DS1307 rtc;                // Real-time clock
LiquidCrystal lcd(7, 8, 9, 10, 11, 12); // LCD display

// Camera serial communication
SoftwareSerial cameraSerial(CAMERA_RX, CAMERA_TX);

// Variables for face recognition
String recognizedPatientId = "";
bool faceRecognized = false;

// Struct to store patient data
struct Patient {
  String id;
  String name;
  String photoUrl;
  String pillTimes;
  int slotNumber;
};

// Array to store patients from the database
const int MAX_PATIENTS = 10;
Patient patients[MAX_PATIENTS];
int patientCount = 0;

// Function prototypes
void connectToWiFi();
void fetchPatientData();
bool checkPatientAccess(String patientId);
void rotateServoToSlot(int slotNumber);
void displayMessage(String line1, String line2, int duration);
void processFaceRecognition();
void displayDateTime();

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  cameraSerial.begin(115200);
  
  // Initialize LCD
  lcd.begin(16, 2);
  displayMessage("Starting up...", "", 2000);
  
  // Initialize other components
  Wire.begin();
  rtc.begin();
  SPI.begin();
  rfid.PCD_Init();
  
  // Set up servo
  pillDispenser.attach(SERVO_PIN);
  pillDispenser.write(0); // Reset position
  
  // Set up LED pins
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  
  // Connect to WiFi
  connectToWiFi();
  
  // Fetch patient data from API
  fetchPatientData();
  
  displayMessage("System Ready", "Waiting for face", 0);
}

void loop() {
  // Display current date and time
  displayDateTime();
  
  // Process face recognition data from camera
  processFaceRecognition();
  
  // If a face was recognized, check if they have access
  if (faceRecognized && recognizedPatientId != "") {
    // Check if patient can access pills at current time
    if (checkPatientAccess(recognizedPatientId)) {
      // Find the patient to get their slot number
      int slotNumber = 0;
      String patientName = "";
      
      for (int i = 0; i < patientCount; i++) {
        if (patients[i].id == recognizedPatientId) {
          slotNumber = patients[i].slotNumber;
          patientName = patients[i].name;
          break;
        }
      }
      
      // Grant access
      digitalWrite(GREEN_LED, HIGH);
      displayMessage("Access Granted", "Hello " + patientName, 2000);
      
      // Move servo to dispense from the correct slot
      rotateServoToSlot(slotNumber);
      
      digitalWrite(GREEN_LED, LOW);
      displayMessage("Pills Dispensed", "Slot: " + String(slotNumber), 3000);
    } else {
      // Deny access - not within pill time window
      digitalWrite(RED_LED, HIGH);
      displayMessage("Access Denied", "Not pill time", 3000);
      digitalWrite(RED_LED, LOW);
    }
    
    // Reset for next recognition
    faceRecognized = false;
    recognizedPatientId = "";
  }
  
  // Check for manual commands via Serial
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "REFRESH") {
      displayMessage("Refreshing Data", "Please wait...", 1000);
      fetchPatientData();
      displayMessage("Data Updated", "", 2000);
    } else if (command.startsWith("TEST_ID:")) {
      // For testing: TEST_ID:patient_id
      String testId = command.substring(8);
      recognizedPatientId = testId;
      faceRecognized = true;
    }
  }
  
  delay(100); // Small delay for stability
}

// Connect to WiFi network
void connectToWiFi() {
  displayMessage("Connecting to", ssid, 0);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    lcd.setCursor(attempts % 16, 1);
    lcd.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    displayMessage("WiFi Connected", WiFi.localIP().toString(), 2000);
  } else {
    displayMessage("WiFi Failed!", "Check settings", 2000);
  }
}

// Fetch patient data from the API
void fetchPatientData() {
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
    if (WiFi.status() != WL_CONNECTED) {
      return; // Still not connected
    }
  }
  
  HTTPClient http;
  String url = String(apiBaseUrl) + "/patients";
  
  http.begin(url);
  int httpCode = http.GET();
  
  if (httpCode == HTTP_CODE_OK) {
    String payload = http.getString();
    
    // Parse JSON response
    DynamicJsonDocument doc(4096); // Adjust size as needed
    DeserializationError error = deserializeJson(doc, payload);
    
    if (!error) {
      patientCount = 0;
      JsonArray array = doc.as<JsonArray>();
      
      for (JsonObject patient : array) {
        if (patientCount < MAX_PATIENTS) {
          patients[patientCount].id = patient["id"].as<String>();
          patients[patientCount].name = patient["name"].as<String>();
          patients[patientCount].photoUrl = patient["photo"].as<String>();
          patients[patientCount].pillTimes = patient["pillTimes"].as<String>();
          patients[patientCount].slotNumber = patient["slotNumber"].as<int>();
          patientCount++;
        }
      }
      
      Serial.print("Loaded ");
      Serial.print(patientCount);
      Serial.println(" patients from database");
    } else {
      Serial.print("JSON parsing error: ");
      Serial.println(error.c_str());
    }
  } else {
    Serial.print("HTTP error: ");
    Serial.println(httpCode);
  }
  
  http.end();
}

// Check if a patient has access at the current time
bool checkPatientAccess(String patientId) {
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
    if (WiFi.status() != WL_CONNECTED) {
      return false; // Still not connected
    }
  }
  
  HTTPClient http;
  String url = String(apiBaseUrl) + "/access/" + patientId;
  
  http.begin(url);
  int httpCode = http.GET();
  
  if (httpCode == HTTP_CODE_OK) {
    String payload = http.getString();
    
    // Parse JSON response
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, payload);
    
    if (!error) {
      bool access = doc["access"];
      http.end();
      return access;
    }
  }
  
  http.end();
  return false;
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

// Process face recognition data from camera
void processFaceRecognition() {
  // In a real implementation, this would receive data from the camera module
  // For this example, we'll simulate face recognition using RFID as a stand-in
  
  // Check if a new RFID card is present
  if (rfid.PICC_IsNewCardPresent() && rfid.PICC_ReadCardSerial()) {
    // Read UID
    String cardUID = "";
    for (byte i = 0; i < rfid.uid.size; i++) {
      cardUID += String(rfid.uid.uidByte[i], HEX);
    }
    
    // In a real implementation, we would match faces against photos from the database
    // For this demo, we'll match RFID UIDs to patient IDs by taking first patient
    if (patientCount > 0) {
      recognizedPatientId = patients[0].id;
      faceRecognized = true;
      
      Serial.print("Recognized patient ID: ");
      Serial.println(recognizedPatientId);
    }
    
    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
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