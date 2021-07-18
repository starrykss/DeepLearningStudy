"""
Title : FC Layers Practice with MNIST Dataset
Date : July 18, 2021
Author : starrykss
"""

# 0. Import Library
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## 1.1 Data Analysis
print(type(train_images))

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

plt.imshow(train_images[0], cmap='gray')
plt.title('MNIST digit')
plt.xlabel('number is 5')
plt.ylabel('28x28 image')
plt.xticks([])
plt.yticks([])
plt.show()

## 1.2 Data Normalization : Pixel range 0 to 255 -> 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# 2. Model
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))     # softmax는 마지막에 사용

model.summary()

# 3. Train
## 3.1 Train (How to train data?)
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

## 3.2 Train (Training Data)
history = model.fit(train_images, train_labels, batch_size=128, epochs=10)

# 4. Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)

## 0th tested image prediction probability
print("Probability", prediction[0], sep='\n')

## 0th tested image prediction result
pred = np.argmax(prediction[0])
print("result: ", pred)

## 0th true
label = test_labels[0]
print("True label: ", label)

## Error Analysis
incorrect = []
for i in range(10000):
  if np.argmax(prediction[i]) != test_labels[i]:
    incorrect.append(i)

print(incorrect)
print(len(incorrect))

plt.subplot(1, 2, 1)
plt.imshow(test_images[61])
plt.xticks([])
plt.yticks([])
plt.xlabel(f"True: {test_labels[61]}")

plt.subplot(1, 2, 2)
plt.stem(prediction[61])

plt.show()

loss = history.history['loss']
acc = history.history['accuracy']

plt.plot(range(1, 11), loss)
plt.title('Loss values according to epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.show()

plt.plot(range(1, 11), acc)
plt.title('Accuracy values according to epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy vakue')
plt.show()