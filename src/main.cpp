#include <Arduino.h>
#include <Wire.h>
#include <RTClib.h>
#include <LiquidCrystal.h>
#include <Servo.h>
#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 53   // RC522 SDA (Slave Select) on pin 25
#define RST_PIN 3   // RC522 Reset
#define GREEN_LED 4 // Green LED for correct card
#define RED_LED 5   // Red LED for incorrect card

MFRC522 rfid(SS_PIN, RST_PIN); // Create MFRC522 instance

byte authorizedUID[] = { 0x37, 0x8F, 0xF8, 0x0 }; 

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
  SPI.begin();        // Initialize SPI bus
  rfid.PCD_Init();    // Initialize RFID scanner

  Serial.println("RFID Scanner Ready...");
  Serial.println("Scan your card/tag...");

  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  if (!rtc.isrunning()) {
    lcd.print("RTC not running!");
    // Set time to compile time:
    // rtc.adjust(DateTime(F(DATE), F(TIME)));
  }
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

  lcd.setCursor(0, 0);
  lcd.print("Time:");

  // Format hour
  if (adjustedHour < 10) lcd.print("0");
  lcd.print(adjustedHour);
  lcd.print(":");

  // Format minute
  if (adjustedMinute < 10) lcd.print("0");
  lcd.print(adjustedMinute);

  lcd.setCursor(0, 1);
  lcd.print("Date:");
  lcd.print(now.month());
  lcd.print("/");
  if (adjustedHour > 12) lcd.print(now.day() - 1);
  else lcd.print(now.day());
  lcd.print("/");
  lcd.print(now.year());

  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "UNLOCK") {
      rotateServo();
      lcd.clear(); // Clear the screen before printing new text
      lcd.setCursor(0, 0);
      lcd.print("Ready to");
      lcd.setCursor(0, 1);
      lcd.print("Dispense!");
      delay(5000);
      lcd.clear();
      lcd.setCursor(0, 0);
    }
    if (command == "LOCK") {
      lcd.clear(); // Clear the screen before printing new text
      lcd.setCursor(0, 0);
      lcd.print("Locked!");
      delay(5000);
      lcd.clear();
      lcd.setCursor(0, 0);
    }
  }

  // Only check RFID if a new card is available
  if (rfid.PICC_IsNewCardPresent() && rfid.PICC_ReadCardSerial()) {
    Serial.print("Scanned UID: ");
    bool isAuthorized = true;

    for (byte i = 0; i < rfid.uid.size; i++) {
      Serial.print(rfid.uid.uidByte[i], HEX);
      Serial.print(" ");

      if (rfid.uid.uidByte[i] != authorizedUID[i]) {
        isAuthorized = false;
      }
    }
    Serial.println();

    if (isAuthorized) {
      Serial.println("✅ Access Granted!");
      digitalWrite(GREEN_LED, HIGH);
      delay(2000); // Display LED for 2 seconds
      digitalWrite(GREEN_LED, LOW);
    } else {
      Serial.println("❌ Access Denied!");
      digitalWrite(RED_LED, HIGH);
      delay(2000); // Display LED for 2 seconds
      digitalWrite(RED_LED, LOW);
    }

    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
  }

  // Reduce delay to make the loop more responsive
  // delay(50);  // Reduced delay for faster processing of new cards
}
