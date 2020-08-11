#include<Keyboard.h>
int led = 13;
void setup() {
 // open the serial port:
 Serial.begin(19200); 
 pinMode(led, OUTPUT);
}

void loop() {
 // check for incoming serial data:
 if (Serial.available() > 0) {
  // read incoming serial data:
  char inChar = Serial.read();
  if(inChar == 'w') 
  {
   digitalWrite(led, HIGH);
  }
  else if(inChar == 'a')
  {
   digitalWrite(led, LOW); 
  }
 }
}
