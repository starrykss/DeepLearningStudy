"""
Title : Linear Regression Practice with Boston Housing Dataset
Date : July 13, 2021
Author : starrykss
"""

# Load Library(Package)
from tensorflow.keras import datasets
from sklearn.linear_model import LinearRegression

# Load Data
(train_data, train_label), (test_data, test_label) = datasets.boston_housing.load_data()

# Train
model_lr = LinearRegression()
model_lr.fit(train_data, train_label)

# Evaluation
prediction_lr = model_lr.predict(test_data)
error = sum(abs(prediction_lr - test_label)) / len(test_label)
print(error)