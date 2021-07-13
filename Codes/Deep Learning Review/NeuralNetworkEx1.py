"""
Title : Neural Network Practice with Breast Cancer Dataset
Date : July 13, 2021
Author : starrykss
"""

# Load Library(Package)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Load Data
cancer = load_breast_cancer()
data = cancer['data']
target = cancer['target']
train_data, test_data, train_label, test_label = train_test_split(data, target, test_size=0.2, random_state=1)

# Model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(30,)),   # 30 features
    keras.layers.Dense(1, activation='sigmoid')
])

# How to train the model?
model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(train_data, train_label, batch_size=16, epochs=20)    
# Ex. Data 10000개
# batch : 16개씩 훈련
# epochs : 10000/16 = 약 600번, 이 과정을 20번 반복

# Evaluation
loss, acc = model.evaluate(test_data, test_label)
print(acc)