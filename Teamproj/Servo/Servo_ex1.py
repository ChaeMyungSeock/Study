# python3
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)
import time
for i in range(60,120,30):
	time.sleep(0.5)
	for r in range(3):
		kit.servo[r].angle=i
		time.sleep(0.5)