"""
Title : CNN(Convolutional Neural Network) Practice using Zero Padding with Fashion MNIST Dataset
Date : July 18, 2021
Author : starrykss
"""

# 0. Import Library
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
fashion = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

## 1.1 Data Analysis
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

plt.imshow(train_images[0], cmap='gray')
plt.show()

## 1.2 Data Normalization
train_images = train_images / 255.0
test_images = test_images / 255.0

mean_image = np.mean(train_images)
print(mean_image)

std_image = np.std(train_images)
print(std_image)

train_images = (train_images - mean_image) / std_image
test_images = (test_images - mean_image) / std_image

print(np.mean(train_images))
print(np.std(train_images))

## 1.3 Data Reshape
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# 2. Model
model = keras.Sequential() 
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))   # Zero Padding
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))                                        
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))                            # Zero Padding
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))                                           
# Flatten and FC Layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))

model.summary()

# 3. Train
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=128, epochs=10)

# 4. Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)

loss = history.history['loss']
acc = history.history['accuracy']

plt.plot(loss)
plt.title('Loss values according to epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.show()

plt.plot(acc)
plt.title('Accuracy values according to epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy value')
plt.show()