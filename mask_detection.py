import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
last_beep_time = 0

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import winsound

# Load trained model
model = load_model("face_mask_model.h5")

# Load DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt",   # Make sure NO .txt at end
    "res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
img_size = 128

print("Face Mask Detection Started... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Detect faces using DNN
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence_face = detections[0, 0, i, 2]

        if confidence_face > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Resize to model input size
            face = cv2.resize(face, (img_size, img_size))

            # SAME preprocessing as training
            face = face / 255.0

            # Add batch dimension
            face = np.reshape(face, (1, img_size, img_size, 3))

            # Predict
            prediction = model.predict(face, verbose=0)[0][0]

            # Class mapping:
            # with_mask = 0
            # without_mask = 1

            if prediction > 0.6:
                label = "No Mask"
                color = (0, 0, 255)
                confidence = prediction * 100
                winsound.Beep(1000, 200)
                
                if time.time() - last_beep_time > 2:
                    winsound.Beep(1000, 200)
                    last_beep_time = time.time()
            else:
                label = "Mask"
                color = (0, 255, 0)
                confidence = (1 - prediction) * 100

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Put text
            cv2.putText(frame,
                        f"{label} ({confidence:.2f}%)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()