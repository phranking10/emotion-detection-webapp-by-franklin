import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# path to your dataset
dataset_path = r"C:\Users\User\Documents\EBOAGWU_23CG034057_EMOTION_DETECTION_WEB_APP\emotion_dataset"

# preprocess
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(48,48),
    batch_size=8,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(48,48),
    batch_size=8,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')   # 4 emotions
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# train
model.fit(train_data, epochs=10, validation_data=val_data)

# save model
model.save("emotion_model.h5")

print("✅ Model training complete & saved as emotion_model.h5")
