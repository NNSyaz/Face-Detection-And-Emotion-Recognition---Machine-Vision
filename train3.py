# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 01:10:25 2023

@author: syazw
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the path to your dataset
dataset_path = r'C:\Users\syazw\Desktop\UMP\sem5\BTI3423 MACHINE VISION\Project\combined'

# Constants for image size and batch size
img_width, img_height = 48, 48
batch_size = 50

# Create an ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.20)

# Load and prepare the training data
train_generator = datagen.flow_from_directory(
    dataset_path + '/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

# Load and prepare the validation data
validation_generator = datagen.flow_from_directory(
    dataset_path + '/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Define and compile the model
model = Sequential() #Creates linear stack of layers for the model
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))  # Output layer with 8 classes for emotions

#Configures the model for training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# Load and prepare the test data
test_generator = datagen.flow_from_directory(
    dataset_path + '/test',  
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test Accuracy after further augmentation: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save('trained_emotion_model3_2.keras')

