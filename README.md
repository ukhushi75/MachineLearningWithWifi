# 📶 Human Activity Detection using CSI Data

In this project, I used **Python** and various machine learning models to detect human activity from CSI (Channel State Information) packet data collected via Wi-Fi router antennas.

## 🏃 Activities Detected
Each activity is represented by an integer label:
- 1: Lie down
- 2: Fall
- 3: Walk
- 4: Pick up
- 5: Run
- 6: Sit down
- 7: Stand up

## 🎯 Goal
To recognize different human movements based on signal data and evaluate classifier accuracy and performance.

## 🤖 Models Implemented
- **K-Nearest Neighbors (KNN)** using different values of `k` (1, 2, 5, 10, 15)
- **Support Vector Machine (SVM)** with default hyperparameters
- **Random Forest Classifier** with 5, 10, and 20 estimators

## 📊 Evaluation
- **Accuracy** scores were calculated for all models.
- **Normalized Confusion Matrices** were plotted to visualize classification performance and misclassifications.
- Models were compared based on both accuracy and computational complexity.

## 📁 Dataset Info
- Shape of training data: `(3977, 250, 90)`
- Shape of test data: `(500, 250, 90)`
- Labels: scalar values for each activity

> 🛠️ Data reshaping was necessary before training, as machine learning models require 1D input features.
