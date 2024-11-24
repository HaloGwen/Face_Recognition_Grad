import cv2
import os

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

raw_folder = 'dataset/raw'
processed_folder = 'dataset/processed'

for class_folder in os.listdir(raw_folder):
    class_path = os.path.join(raw_folder, class_folder)

    if os.path.isdir(class_path):
        print(f"Processing folder: {class_folder}")
        
        processed_class_path = os.path.join(processed_folder, class_folder)
        if not os.path.exists(processed_class_path):
            os.makedirs(processed_class_path)

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Warning: Cannot read file {image_path}")
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = detector.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cropped_face = gray[y:y+h, x:x+w]
                output_path = os.path.join(processed_class_path, image_file)
                cv2.imwrite(output_path, cropped_face)
                print(f"Saved: {output_path}")
