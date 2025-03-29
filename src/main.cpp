#include <Arduino.h>
#include <Wire.h>
#include <RTClib.h>
#include <LiquidCrystal.h>
#include <Servo.h>

#define GREEN_LED 4 // Green LED for correct face recognition
#define RED_LED 5   // Red LED for incorrect face

Servo myServo;

RTC_DS1307 rtc;
LiquidCrystal lcd(7, 8, 9, 10, 11, 12);

void rotateServo() {
  for (int pos = 0; pos <= 180; pos += 5) {
    myServo.write(pos);
    delay(20);
  }
  delay(500); // Pause at 180
  for (int pos = 180; pos >= 0; pos -= 5) {
    myServo.write(pos);
    delay(20);
  }
}

void setup() {
  lcd.begin(16, 2);
  Wire.begin();
  rtc.begin();
  myServo.attach(13);

  Serial.begin(9600); // Start Serial Monitor

  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  if (!rtc.isrunning()) {
    lcd.print("RTC not running!");
    // Set time to compile time:
    // rtc.adjust(DateTime(F(DATE), F(TIME)));
  }
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Camera System");
  lcd.setCursor(0, 1);
  lcd.print("Ready...");
  
  Serial.println("Camera Face Recognition System Ready");
}

void loop() {
  DateTime now = rtc.now();

  // Adjusting time by 12 hours and 3 minutes
  int adjustedHour = now.hour() + 12;
  int adjustedMinute = now.minute() + 3;

  if (adjustedMinute >= 60) {
    adjustedMinute -= 60;
    adjustedHour++;
  }

  if (adjustedHour >= 24) {
    adjustedHour -= 24;
  }

  // Only check Serial if data is available
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "UNLOCK") {
      // Face recognized - unlock
      rotateServo();
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Face Recognized");
      lcd.setCursor(0, 1);
      lcd.print("Dispensing...");
      
      digitalWrite(GREEN_LED, HIGH);
      delay(2000);
      digitalWrite(GREEN_LED, LOW);
      
      delay(5000);
      
      // Return to default display
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Time: ");
      if (adjustedHour < 10) lcd.print("0");
      lcd.print(adjustedHour);
      lcd.print(":");
      if (adjustedMinute < 10) lcd.print("0");
      lcd.print(adjustedMinute);
    }
    else if (command == "LOCK") {
      // Unrecognized face - keep locked
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Access Denied!");
      lcd.setCursor(0, 1);
      lcd.print("Unknown Face");
      
      digitalWrite(RED_LED, HIGH);
      delay(2000);
      digitalWrite(RED_LED, LOW);
      
      delay(5000);
      
      // Return to default display
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Time: ");
      if (adjustedHour < 10) lcd.print("0");
      lcd.print(adjustedHour);
      lcd.print(":");
      if (adjustedMinute < 10) lcd.print("0");
      lcd.print(adjustedMinute);
    }
  }
  
  // Update time display every 30 seconds
  static unsigned long lastDisplay = 0;
  if (millis() - lastDisplay >= 30000) {
    lastDisplay = millis();
    
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Time: ");
    if (adjustedHour < 10) lcd.print("0");
    lcd.print(adjustedHour);
    lcd.print(":");
    if (adjustedMinute < 10) lcd.print("0");
    lcd.print(adjustedMinute);
    
    lcd.setCursor(0, 1);
    lcd.print("Date: ");
    lcd.print(now.month());
    lcd.print("/");
    if (adjustedHour > 12) lcd.print(now.day() - 1);
    else lcd.print(now.day());
  }
}