"""
Title : Logistic Regression Practice with Breast Cancer Dataset
Date : July 13, 2021
Author : starrykss
"""

# Load Library(Package)
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Data
cancer = load_breast_cancer()
data = cancer['data']
target = cancer['target']
train_data, test_data, train_label, test_label = train_test_split(data, target, test_size=0.2, random_state=1)

# Train
model_lr = LogisticRegression()
model_lr.fit(train_data, train_label)

# Evaluation
prediction_lr = model_lr.predict(test_data)
accuracy = sum(abs(prediction_lr == test_label)) / len(test_label)
print(accuracy)