# Face Mask Detection System with Live Alert System

## Overview
This project implements a real-time face mask detection system using deep learning and computer vision. The system detects faces through a webcam and classifies whether the person is wearing a mask or not. If a person is detected without a mask, the system generates an audio alert.

The model is built using MobileNetV2 transfer learning and deployed using OpenCV for real-time detection.

---

## Features
- Real-time face detection using webcam  
- Face mask classification using deep learning  
- Audio alert when a person is not wearing a mask  
- Bounding box with confidence score  
- Lightweight and fast model  

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- MobileNetV2 (Transfer Learning)  
- OpenCV  
- NumPy  
- Winsound (Windows alert sound)  

---

## Dataset Structure
'''dataset/
│
├── with_mask/
│ ├── image1.jpg
│ ├── image2.jpg
│
└── without_mask/
├── image1.jpg
├── image2.jpg'''


---

## Model Architecture
The system uses MobileNetV2 as a feature extractor.

- Base Model: MobileNetV2 (pretrained on ImageNet)  
- Global Average Pooling Layer  
- Dense Layer  
- Dropout Layer  
- Output Layer (Sigmoid activation)  

Loss Function: Binary Crossentropy  
Optimizer: Adam  

---

## Training Details
- Image Size: 128 x 128  
- Batch Size: 32  
- Epochs: 10  
- Validation Split: 20%  

### Data Preprocessing
Images are normalized using:
rescale = 1./255

---

## Face Detection Method
The project uses OpenCV DNN Face Detector:

- deploy.prototxt  
- res10_300x300_ssd_iter_140000.caffemodel  

This method works better than Haar Cascade when faces are partially covered.

---

## How the System Works
1. Webcam captures real-time frames  
2. OpenCV detects faces  
3. Faces are resized to 128x128  
4. Model predicts mask status  
5. Display output:
   - Green box → Mask  
   - Red box → No Mask  
6. Alert sound is triggered for No Mask  

---

## Installation

Install required libraries:

pip install tensorflow opencv-python numpy

---

## Project Files

face_mask_model.h5  
mask_detection.py  
Face_Mask_Detection.ipynb
Video Project.mp4

---

## Run the Project

python mask_detection.py

Press **Q** to exit the webcam.

---

## Performance
- Training Accuracy: ~97–98%  
- Validation Accuracy: ~97%  
- Real-time detection supported  

---

## Future Improvements
- Deploy as desktop application  
- Add face recognition  
- Use Raspberry Pi for edge deployment  
- Integrate with CCTV systems  

---

## Author
Parth Gupta
Elevate Labs
