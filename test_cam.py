import cv2

print("Camera check shuru ho raha hai...")
# CAP_DSHOW direct Windows ke camera driver se connect karta hai
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera access nahi ho raha! Check karo ke koi aur app toh camera use nahi kar rahi.")
else:
    print("Success: Camera mil gaya! Window close karne ke liye 'q' dabao.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read nahi hua.")
            break
        cv2.imshow("Test Window", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()