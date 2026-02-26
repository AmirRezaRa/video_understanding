import os 
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2
from keras.layers import Dense , Conv3D, MaxPool3D, Flatten, Dropout, BatchNormalization 
from keras.models import Sequential
from keras.optimizers import Adam


# import kagglehub
# path = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")
# print("Path to dataset files:", path)


IMG_SIZE = 128
SEQUENCE_LENGTH = 16
DATASET_DIR = 'D:/main_project_vi/face/video_understanding/dataset'
CLASSES = ['NoneViolence', 'Violence']

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    skip = max(total_frames // SEQUENCE_LENGTH , 1)
    
    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*skip)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
    
    cap.release()
    
    if len(frames)< SEQUENCE_LENGTH:
        frames.append(frames[-1])
    
    return np.array(frames)

# creat data

X = []
Y = []

for lable, class_name in enumerate(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)

    for video in os.listdir(class_path):
        video_path = os.path.join(class_path, class_name)
        frames = extract_frames(video_path)
        
        if len(frames) == SEQUENCE_LENGTH:
            X.append(frames)
            Y.append(lable)

X = np.array(X)
Y = np.array(Y)


# train_test_split/creat model

x_train,x_test, y_train,y_test = train_test_split(X, Y , test_size=0.2, random_state=42)


model = Sequential([
    Conv3D(32, kernel_size=(3,3,3), activation='relu', input_shape = (SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE,3)),
    BatchNormalization(),
    MaxPool3D(),
    
    Conv3D(64, kernel_size=(3,3,3), activation='relu'),
    BatchNormalization(),
    MaxPool3D(),
    
    Conv3D(128, kernel_size=(3,3,3), activation='relu'),
    BatchNormalization(),
    MaxPool3D(),
    
    Flatten(),
    Dense(256, activation= 'relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam('1e-4'),
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_split=0.2, epochs=15, batch_size=2
)

loss, acc = model.evaluate(x_test, y_test)
print('test accuracy:', acc)

model.save('video_understanding_model.h5')