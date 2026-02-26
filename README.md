## Real-Time Violence Detection Using 3D‑CNN
## Overview
This project implements a real-time violence detection system using 3D Convolutional Neural Networks (3D‑CNN). Unlike traditional image-based models, this model processes sequences of video frames to capture temporal patterns, enabling accurate detection of violent events in videos.
The system can run on a CPU, making it accessible for personal laptops, and it includes a real-time demo using a webcam.


## Features
Detects violent vs non-violent activities in videos.
Real-time detection with live webcam feed.
Lightweight 3D‑CNN suitable for CPU execution.
Preprocessing of video frames including resizing and normalization.
Easy to extend with datasets like Real Life Violence Dataset or UCF‑Crime.

## Dataset
This project is compatible with:
Real Life Violence Situations Dataset – lightweight and easy to use for CPU projects.
UCF-Crime Dataset – optional, for more complex and real-world scenarios.

## Installation

Install required packages:
pip install tensorflow opencv-python numpy scikit-learn

## Usage
for train:
    python train_3D_CNN.py

for running real_time detection:
    python real_time_3D_CNN.py


## Model Architecture
3D‑CNN:
    Conv3D + MaxPool3D layers to capture spatio-temporal patterns.
    Flatten + Dense layers for classification.
    Sigmoid activation for binary classification.
    Input: (SEQUENCE_LENGTH=16, IMG_SIZE=128, 128, 3)


## Parameters
SEQUENCE_LENGTH = 16
Batch Size = 2 (for CPU training)
Optimizer: Adam (lr=1e-4)
Loss: Binary Crossentropy


## Output example:

Violence (0.87)   red text
Non-Violence (0.12)   green text
