#include <Arduino.h>
#include <Wire.h>
#include <RTClib.h>
#include <LiquidCrystal.h>
#include <Servo.h>
#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 53   // RC522 SDA (Slave Select) on pin 25
#define RST_PIN 2  // RC522 Reset
#define GREEN_LED 4 // Green LED for correct card
#define RED_LED 3   // Red LED for incorrect card

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

  lcd.setCursor(0, 0);
  lcd.print("Time:");
  if (now.hour() > 12) lcd.print(now.hour() - 12);
  if (now.hour() < 12) lcd.print(now.hour() + 12); 
  lcd.print(":");
  if (now.minute() + 3 < 10) lcd.print("0");
  if (now.minute() + 3 == 60) lcd.print("00");
  if (now.minute() + 3 == 61) lcd.print("01");
  if (now.minute() + 3 == 62) lcd.print("02");
  if (now.minute() + 3 == 63) lcd.print("03");
  if (now.minute() + 3 != 60 && now.minute() + 3 != 61 && now.minute() + 3 != 62 && now.minute() + 3 != 63) lcd.print(now.minute() + 3);

  lcd.setCursor(0, 1);
  lcd.print("Date:");
  lcd.print(now.month());
  lcd.print("/");
  lcd.print(now.day());
  lcd.print("/");
  lcd.print(now.year());

  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "UNLOCK") {
      rotateServo();
    }
  }

  if (!rfid.PICC_IsNewCardPresent() || !rfid.PICC_ReadCardSerial()) {
    return; // No card detected
  }

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
    delay(2000);
    digitalWrite(GREEN_LED, LOW);
  } else {
    Serial.println("❌ Access Denied!");
    digitalWrite(RED_LED, HIGH);
    delay(2000);
    digitalWrite(RED_LED, LOW);
  }

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();

  while(rfid.PICC_IsNewCardPresent()) {
    delay(100);
  }

  delay(1000);
}