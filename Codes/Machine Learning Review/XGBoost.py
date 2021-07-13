"""
Title : XGBoost Practice with Breast Cancer Dataset
Date : July 13, 2021
Author : starrykss
"""

# Load Library(Package)
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load Data
cancer = load_breast_cancer()
data = cancer['data']
target = cancer['target']
train_data, test_data, train_label, test_label = train_test_split(data, target, test_size=0.2, random_state=1)

# Train
model_xg = xgb.XGBClassifier()
model_xg.fit(train_data, train_label)

# Evaluation
prediction_xg = model_xg.predict(test_data)
accuracy = sum(abs(prediction_xg == test_label)) / len(test_label)
print(accuracy)