import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

IMG_SIZE = 128
SEQUENCE_LENGTH = 16

model = load_model('video_understanding_model.h5')

cap = cv2.VideoCapture(0)

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_norm = frame_resized / 255.0
    frames.append(frame_norm)
    
    if len(frames) > SEQUENCE_LENGTH:
        frames.pop(0)
    
    if len(frames) == SEQUENCE_LENGTH:
        input_batch = np.expand_dims(np.array(frames), axis=0)
        pred= model.predict(input_batch, verbose=0)[0][0]
        lable = "Violence"if pred > 0.5 else 'NoneViolence'
        color = (0,0,255) if lable == "Violence" else (0,255,0)
        
        cv2.putText(frame, f"{lable} ({pred:.2f})", (10,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, color, 2)
    
    cv2.imshow('real_time violence detection', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()