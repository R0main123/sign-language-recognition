import cv2
import numpy as np
from tensorflow.keras.models import load_model


model_path = 'hand_recognition_model.h5'
model = load_model(model_path)

alphabet_mapper = {i: chr(65 + i) for i in range(26)}

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0
    return img_normalized

def predict_hand_gesture(img_processed):
    img_expanded = np.expand_dims(img_processed, axis=(0, -1))
    output = model.predict(img_expanded, verbose=0)
    predicted_index = np.argmax(output)
    confidence = output[0][predicted_index]
    return predicted_index, confidence

cap = cv2.VideoCapture(0)

x, y, w, h = 800, 100, 300, 300

last_prediction = ""
display_until = 0
confidence_threshold = 0.8

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = frame[y:y + h, x:x + w]
    img_processed = process_image(roi)


    predicted_index, confidence = predict_hand_gesture(img_processed)

    if confidence > confidence_threshold:
        last_prediction = alphabet_mapper[predicted_index]
        display_until = cv2.getTickCount() + cv2.getTickFrequency() * 2  # Afficher 2 secondes

    if cv2.getTickCount() < display_until:
        cv2.putText(frame, f"Prediction: {last_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()