import cv2
import tensorflow as tf
import numpy as np
import os

# Constants
MODEL_PATH = "face_recognition.h5"
IMAGE_PATH = 'nam.jpg'
IMG_SIZE = (100, 100)
DATASET_PATH = './dataset/processed/'
THRESHOLD = 0.8 

LABELS = sorted(os.listdir(DATASET_PATH))

model = tf.keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, IMG_SIZE)
    roi_gray = roi_gray.astype('float32') / 255.0  # Normalize to [0, 1]
    roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
    roi_gray = np.expand_dims(roi_gray, axis=0)   # Add batch dimension

    prediction = model.predict(roi_gray)
    confidence = np.max(prediction)  # Lấy giá trị xác suất cao nhất
    predicted_label = np.argmax(prediction)

    if confidence >= THRESHOLD:
        label_name = LABELS[predicted_label]
    else:
        label_name = "Unknown"

    display_text = f"{label_name} ({confidence:.2f})"
    
    print(f'name: {label_name}, confidence: {confidence}')