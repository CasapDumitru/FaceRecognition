#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Firmata.h>

LiquidCrystal_I2C lcd(0x27,16,2);
int lastLine = 1;
int led1 = 8;
int led2 = 9;
int led3 = 10;

void stringDataCallback(char *stringData){

   lcd.clear();
   
   String str;
   str.concat(stringData);
   str = str.substring(0, str.length()-1);
  
   int cursorIndex = (16 - str.length()) / 2; 
   
   if(str.equals("Waiting")) {
     digitalWrite(led1,LOW);
     digitalWrite(led2,LOW);
     digitalWrite(led3,LOW);
     lcd.setCursor(cursorIndex,0);
     lcd.print(str);
   }
   else if(str.equals("Face Detected")) {
     digitalWrite(led1,LOW);
     digitalWrite(led2,HIGH);
     digitalWrite(led3,LOW);
     lcd.setCursor(cursorIndex,0);
     lcd.print(str);
   } else if(str.equals("NOT RECOGNIZED")) {
     digitalWrite(led1,LOW);
     digitalWrite(led2,LOW);
     digitalWrite(led3,HIGH);
     lcd.setCursor(cursorIndex,0);
     lcd.print(str);
   } else {
     digitalWrite(led1,HIGH);
     digitalWrite(led2,LOW);
     digitalWrite(led3,LOW);
     lcd.setCursor(5,0);
     lcd.print("HELLO");
     str += "!";
     lcd.setCursor(cursorIndex,1);
     lcd.print(str);
   }
   
   
}

void setup() {
  lcd.init();
  lcd.backlight();
  pinMode(led1,OUTPUT);
  pinMode(led2,OUTPUT);
  pinMode(led3,OUTPUT);
  Firmata.setFirmwareVersion( FIRMATA_MAJOR_VERSION, FIRMATA_MINOR_VERSION );
  Firmata.attach( STRING_DATA, stringDataCallback);
  Firmata.begin();  
}

void loop() {
  while ( Firmata.available() ) {
    Firmata.processInput();
  }
}
