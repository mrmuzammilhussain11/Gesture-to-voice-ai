import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import time

# 1. Naya model load karo
with open('model.p', 'rb') as f:
    model = pickle.load(f)

# 2. MediaPipe Setup
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Variables for stability
last_prediction = "Analyzing..."
confidence_threshold = 0.50  # 50% se zyada sure hone par result dikhayega

print("AI Testing Shuru! Camera open ho raha hai...")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Prediction
            prediction_proba = model.predict_proba([landmarks])
            max_proba = np.max(prediction_proba)
            
            # Agar 50% se zyada confidence ho
            if max_proba > confidence_threshold:
                last_prediction = model.classes_[np.argmax(prediction_proba)]
                color = (0, 255, 0) # Green for detected
                text_display = f"{last_prediction} ({int(max_proba*100)}%)"
            else:
                color = (0, 0, 255) # Red for low confidence
                text_display = "Analyzing..."

            # UI box
            cv2.rectangle(frame, (0, 0), (640, 50), (30, 30, 30), -1)
            cv2.putText(frame, text_display, (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()