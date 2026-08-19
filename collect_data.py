import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import time

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

TARGET_LIMIT = 500

def count_saved_data(gesture_name):
    if not os.path.exists('data.csv'): return 0
    count = 0
    with open('data.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == gesture_name:
                count += 1
    return count

# Main Loop taake program band na ho
while True:
    label = input("\nKonse ishara record karna hai? (Ya band karne ke liye 'exit' likhen): ")
    
    if label.lower() == 'exit':
        print("Program khatam ho raha hai. Allah Hafiz!")
        break

    current_count = count_saved_data(label)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    print(f"--- Recording '{label}'. Target: {TARGET_LIMIT}. Current: {current_count} ---")
    print("Camera window par ja kar 'q' dabayen agar beech mein rukna ho.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        # UI Display
        cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Count: {current_count}/{TARGET_LIMIT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if results.hand_landmarks:
            if current_count < TARGET_LIMIT:
                cv2.putText(frame, "Saving...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                for hand_landmarks in results.hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    with open('data.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([label] + landmarks)
                    
                    current_count += 1
                    time.sleep(0.05) # Speed thodi tez kar di hai
            else:
                cv2.putText(frame, "DONE! Press 'q' for Next Gesture", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Live Counter", frame)
        
        # 'q' dabane par camera band hoga aur wapas 'input' par jayega
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{label} ka kaam poora! Agla batao...")