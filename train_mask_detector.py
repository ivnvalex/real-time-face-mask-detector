#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""A script to train a face mask detection model.
Built on TensorFlow, Keras, OpenCV.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if __name__ == '__main__':

    """Initialize parameters."""
    lr = 1e-4  # learning rate
    epochs = 20
    bs = 32  # batch size
    data_dir = os.path.join(os.getcwd(), 'data')
    class_names = ('mask_on', 'mask_off')
    print(f'Initializing parameters:\n'
          f'Learning rate = {lr}\n'
          f'Epochs number = {epochs}\n'
          f'Batch size = {bs}')

    """Initialize images and labels."""
    data = []
    labels = []
    print('Loading images...')
    loading_start_time = datetime.now()
    for class_name in class_names:
        path = os.path.join(data_dir, class_name)
        images_files = os.listdir(path)
        for image_file in images_files:
            image_path = os.path.join(path, image_file)
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)
            labels.append(class_name)

    loading_time = datetime.now() - loading_start_time
    print(f'Images loaded. Loading time = {loading_time}')

    # Labels one-hot encoding
    binarizer = LabelBinarizer()
    labels = binarizer.fit_transform(labels)
    labels = to_categorical(labels)

    data = np.array(data, dtype='float32')
    labels = np.array(labels)

    # Split the data
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels,
        test_size=0.20,
        stratify=labels,
        random_state=0
    )

    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    """Construct a NN based on MobileNetV2."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )

    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(128, activation='relu')(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation='softmax')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    # Freeze each base model layer
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    print('Compiling the model...')
    compiling_start_time = datetime.now()
    opt = Adam(
        lr=lr,
        decay=(lr / epochs)
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    compiling_time = datetime.now() - compiling_start_time
    print(f'Model compiled. Compiling time = {compiling_time}')

    """Train the model."""
    print('Training the model...')
    training_start_time = datetime.now()
    NN = model.fit(
        aug.flow(trainX, trainY, batch_size=bs),
        steps_per_epoch=(len(trainX) // bs),
        validation_data=(testX, testY),
        validation_steps=(len(testX) // bs),
        epochs=epochs
    )
    training_time = datetime.now() - training_start_time
    print(f'Model trained. Training time = {training_time}')

    """Make predictions."""
    print('Proceeding classification...')
    pred_idxs = model.predict(testX, batch_size=bs)
    pred_idxs = np.argmax(pred_idxs, axis=1)

    # Classification report
    print(
        classification_report(
            testY.argmax(axis=1),
            pred_idxs,
            target_names=binarizer.classes_
        )
    )

    """Save the model!"""
    print('Saving the model...')
    model_name = 'mask_detector'
    model.save(
        f'{model_name}.model',
        save_format="h5"
    )
    print(f'Model saved as {model_name}.model')

    # Plot the qualities
    n = epochs
    plt.figure()
    plt.plot(np.arange(0, n), NN.history['loss'], label='Train Loss')
    plt.plot(np.arange(0, n), NN.history['val_loss'], label='Val Loss')
    plt.plot(np.arange(0, n), NN.history['accuracy'], label='Train Accuracy')
    plt.plot(np.arange(0, n), NN.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Accuracy')
    plt.legend(loc='upper right')
    plt.savefig('qualities.png')
    plt.show()
