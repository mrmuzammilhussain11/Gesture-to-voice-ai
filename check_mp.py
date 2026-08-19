import mediapipe as mp
print("MediaPipe version:", mp.__version__)

# Check karein ke 'solutions' module exist karta hai ya nahi
if hasattr(mp, 'solutions'):
    print("SUCCESS: mp.solutions mil gaya!")
else:
    print("ERROR: mp.solutions nahi mil raha. Checking dir()...")
    print(dir(mp))