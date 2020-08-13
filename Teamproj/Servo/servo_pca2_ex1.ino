#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
 
Adafruit_PWMServoDriver pwm1 = Adafruit_PWMServoDriver(0x40);
Adafruit_PWMServoDriver pwm2 = Adafruit_PWMServoDriver(0x41);
#define MIN_PULSE_WIDTH 650
#define MAX_PULSE_WIDTH 2350
#define DEFAULT_PULSE_WIDTH 1500
#define FREQUENCY 50

uint8_t servonum = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("16 channel PWM test!");
 
  pwm1.begin();
  pwm1.setPWMFreq(FREQUENCY);  // This is the maximum PWM frequency
 
  pwm2.begin();
  pwm2.setPWMFreq(FREQUENCY);  // This is the maximum PWM frequency
  }
  int pulseWidth(int angle)
  {
  int pulse_wide, analog_value;
  pulse_wide = map(angle, 0, 180, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  analog_value = int(float(pulse_wide) / 1000000 * FREQUENCY * 4096);
  Serial.println(analog_value);
  return analog_value;
  }



void loop() {
pwm1.setPWM(0, 0, pulseWidth(0));
delay(1000);
pwm2.setPWM(0, 0, pulseWidth(0));
delay(1000);

pwm1.setPWM(0, 0, pulseWidth(120));
delay(500);
pwm2.setPWM(0, 0, pulseWidth(120));
delay(500);
pwm1.setPWM(0, 0, pulseWidth(90));
delay(1000);
pwm2.setPWM(0, 0, pulseWidth(90));
delay(1000);

}
