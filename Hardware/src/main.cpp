#include <Arduino.h>
#include <Wire.h>
#include <RTClib.h>
#include <LiquidCrystal.h>
#include <Servo.h>

Servo myServo;

RTC_DS1307 rtc;
LiquidCrystal lcd(7, 8, 9, 10, 11, 12);

void setup() {
  lcd.begin(16, 2);
  Wire.begin();
  rtc.begin();
  myServo.attach(13);

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
  lcd.print(now.minute() + 3);

  lcd.setCursor(0, 1);
  lcd.print("Date:");
  lcd.print(now.month());
  lcd.print("/");
  lcd.print(now.day());
  lcd.print("/");
  lcd.print(now.year());

  for (int pos = 0; pos <= 180; pos += 5) { // Move from 0° to 180°
    myServo.write(pos);
    delay(20); // Small delay for smooth motion
  }

  for (int pos = 180; pos >= 0; pos -= 5) { // Move back from 180° to 0°
    myServo.write(pos);
    delay(20);
  }

  delay(1000);
}