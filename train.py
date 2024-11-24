import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Activation)
from keras.optimizers import Adam

# Constants
IMG_SIZE = (100, 100)
DATASET_PATH = './dataset/processed/'

# Function to load and preprocess images
def load_dataset(path, img_size):
    data, labels = [], []
    class_names = sorted(os.listdir(path))  # Lấy danh sách các folder lớp, sắp xếp để đảm bảo thứ tự
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(path, class_name)
        if os.path.isdir(class_folder):  # Kiểm tra nếu là thư mục
            for image_file in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_file)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Image {image_path} not found!")
                        continue
                    img = cv2.resize(img, img_size)
                    data.append(img)
                    labels.append(label)  # Gán nhãn theo thứ tự folder
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    data = np.array(data).astype('float32') / 255.0  # Normalize to [0,1]
    data = data.reshape((-1, img_size[0], img_size[1], 1))  # Thêm chiều channel
    labels = np.array(labels)
    return data, labels, len(class_names)  # Trả về data, labels và số lớp (num_classes)

X_train, y_train, NUM_CLASSES = load_dataset(DATASET_PATH, IMG_SIZE)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        Activation("relu"),
        Conv2D(32, (3, 3), padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512),
        Activation("relu"),
        Dense(num_classes),
        Activation("softmax"),
    ])
    return model

# Initialize model
input_shape = (*IMG_SIZE, 1)
model = build_model(input_shape, NUM_CLASSES)
model.summary()

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Train the model
print("Start training...")
model.fit(X_train, y_train, batch_size=5, epochs=10, verbose=1)

# Save the model
model.save("face_recognition.h5")
print("Model saved as 'face_recognition.h5'")
