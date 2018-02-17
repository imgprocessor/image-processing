# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:06:04 2018

@author: User
"""

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO.setup(5, GPIO.IN) #PIR

try:
    time.sleep(2) # to stabilize sensor
    while True:
        if GPIO.input(5):
            print("Motion Detected...")
            time.sleep(1) #to avoid multiple detection
        else:
            print("Not detected...")
            time.sleep(1) #loop delay, should be less than detection delay
        
except:
    GPIO.cleanup()