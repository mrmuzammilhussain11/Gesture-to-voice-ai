import cv2
import pickle
import numpy as np
import pyttsx3
import threading
import os
from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# GLOBAL VARIABLE FOR FRONTEND
latest_prediction = "Waiting..."

# 1. LOAD MODEL & MEDIAPIPE
with open('model.p', 'rb') as f:
    model = pickle.load(f)

model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# 2. VOICE SETUP
voice_lock = threading.Lock()
last_spoken = ""
prediction_history = []  
STABILITY_THRESHOLD = 10 

def speak_text(text):
    with voice_lock:
        try:
            v_engine = pyttsx3.init()
            v_engine.setProperty('rate', 180)
            v_engine.say(text)
            v_engine.runAndWait()
            v_engine.stop()
        except:
            pass

# 3. VIDEO FEED LOGIC
def gen_frames():
    global last_spoken, prediction_history, latest_prediction
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        current_prediction = "Analyzing..."
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            data_aux = []
            for lm in hand_landmarks:
                data_aux.extend([lm.x, lm.y, lm.z])
            try:
                input_data = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(input_data)
                current_prediction = str(prediction[0])
                
                prediction_history.append(current_prediction)
                if len(prediction_history) > STABILITY_THRESHOLD:
                    prediction_history.pop(0)

                if len(prediction_history) == STABILITY_THRESHOLD and all(x == current_prediction for x in prediction_history):
                    if current_prediction != last_spoken:
                        last_spoken = current_prediction
                        if not voice_lock.locked():
                            threading.Thread(target=speak_text, args=(current_prediction,), daemon=True).start()
            except:
                pass

        # UPDATE GLOBAL VARIABLE (Instead of cv2.putText)
        latest_prediction = current_prediction
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# 4. ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# NEW ROUTE FOR FRONTEND DATA
@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": latest_prediction})

@app.route('/search_sign', methods=['POST'])
def search_sign():
    data = request.json
    search_query = data.get('sign', '').upper()
    folder_path = os.path.join('static', 'sign')
    
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            name_part, ext_part = os.path.splitext(filename)
            if name_part.upper() == search_query:
                return jsonify({"success": True, "url": f"/static/sign/{filename}"})
    
    return jsonify({"success": False, "message": "Sign not found in folder"})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)