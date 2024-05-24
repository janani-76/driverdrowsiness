from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import cv2
import os
import pygame
import pyttsx3
import skfuzzy as fuzz
from skfuzzy import control as ctrl

pygame.mixer.init()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]) 
    B = dist.euclidean(eye[2], eye[4]) 
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

EYE_AR_THRESH = 0.25
YAWN_THRESH = 20
COUNTER = 0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Define fuzzy variables and membership functions
eye_ar = ctrl.Antecedent(np.arange(0, 0.5, 0.01), 'eye_ar')
eye_ar['low'] = fuzz.trimf(eye_ar.universe, [0, 0.1, 0.25])
eye_ar['medium'] = fuzz.trimf(eye_ar.universe, [0.2, 0.35, 0.5])
eye_ar['high'] = fuzz.trimf(eye_ar.universe, [0.4, 0.5, 0.5])

alertness = ctrl.Consequent(np.arange(0, 101, 1), 'alertness')
alertness['low'] = fuzz.trimf(alertness.universe, [0, 25, 50])
alertness['medium'] = fuzz.trimf(alertness.universe, [25, 50, 75])
alertness['high'] = fuzz.trimf(alertness.universe, [50, 75, 100])

rule1 = ctrl.Rule(eye_ar['low'], alertness['low'])
rule2 = ctrl.Rule(eye_ar['medium'], alertness['medium'])
rule3 = ctrl.Rule(eye_ar['high'], alertness['high'])

alertness_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
alertness_level = ctrl.ControlSystemSimulation(alertness_ctrl)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        left_eye = roi_gray[10:40, 10:40]
        right_eye = roi_gray[10:40, 40:70]
        
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0
        
        distance = lip_distance(roi_gray)
        
        alertness_level.input['eye_ar'] = ear
        
        alertness_level.compute()
        alert_value = alertness_level.output['alertness']
        
        # Check if the alertness level is low (indicating drowsiness)
        if alert_value < 25:
            # Display drowsiness alert message and play audio alert
            pygame.mixer.music.load('audio/alert.wav')
            pygame.mixer.music.play(-1)
            cv2.putText(frame, "ALERT! You're drowsy", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Check if the eye aspect ratio is below the threshold (indicating closed eyes)
        if ear < EYE_AR_THRESH:
            # Display warning message for closed eyes
            cv2.putText(frame, "Warning: Eyes Closed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Check if yawning is detected
        if distance > YAWN_THRESH:
            # Display yawn alert message
            cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            engine = pyttsx3.init()
            engine.say('Take some fresh air.')
            engine.runAndWait() 

        # Display eye aspect ratio and alertness level on the frame
        cv2.putText(frame, "EYE: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Alertness: {:.2f}".format(alert_value), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Check for key press to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()