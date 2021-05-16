#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""A script to run a real time face mask detection model.
Built on TensorFlow, Keras, OpenCV.
"""

import numpy as np
import cv2
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


def detect_mask(input_frame, face_network, mask_network):
    """Function to detect face and detect mask on a face
    Returns 2-dim tuple with bounding box location and probability of mask on/off
    """

    (h, w) = input_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0)
    )

    # Detect face
    face_network.setInput(blob)
    detections = face_network.forward()

    # Variables initialization
    faces = []
    faces_locations = []
    mask_predictions = []  # Probability

    # Over face detections
    for i in range(0, detections.shape[2]):
        # Get probability
        confidence = detections[0, 0, i, 2]

        # Filter detections
        if confidence > 0.5:
            # Bounding box
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startXpoint, startYpoint, endXpoint, endYpoint) = bbox.astype("int")

            # Make sure the box is in the frame
            (startXpoint, startYpoint) = (max(0, startXpoint), max(0, startYpoint))
            (endXpoint, endYpoint) = (min(w - 1, endXpoint), min(h - 1, endYpoint))

            # Processing the face
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Get faces and their locations
            faces.append(face)
            faces_locations.append((startXpoint, startYpoint, endXpoint, endYpoint))

    # Make sure faces exist
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")

        # Detect a mask
        mask_predictions = mask_network.predict(faces, batch_size=32)

    return (faces_locations, mask_predictions)


if __name__ == '__main__':
    # Load pre-made face detector
    prototxt_path = 'face_detector/deploy.prototxt'
    weights_path = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)

    # Load our trained and saved mask detector
    mask_net = load_model('mask_detector.model')

    # Get camera video stream
    print('Starting live camera video stream...')
    vs = VideoStream(src=0).start()

    # For frames in stream
    while True:
        frame = vs.read()  # Get the frame
        frame = imutils.resize(frame, width=400)

        # Detect faces and make predictions on masks
        (locations, predictions) = detect_mask(frame, face_net, mask_net)

        # For detected faces
        for (box, pred) in zip(locations, predictions):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = 'Mask on' if mask > withoutMask else 'Mask off'  # Prediction label
            color = (0, 255, 0) if label == 'Mask on' else (0, 0, 255)  # Bounding box

            # include the probability in the label
            probability = max(mask, withoutMask) * 100  # Probability of mask on/off
            label = f'{label}: {probability:.2f}%'

            # Display the label and the bounding box on the stream
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow('LIVE video streaming', frame)
        key = cv2.waitKey(1) == ord('q')
