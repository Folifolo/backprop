import numpy as np

def relu(X):
    return X * (X > 0)

def reluD(X):
    return X > 0

#=====================

def calculate_E(label, prediction, size):
    return -np.sum(label * np.log(prediction.T)) / size

def calculate_acc(label, prediction):
    prediction = np.argmax(prediction, axis= 0)
    label = np.argmax(label, axis= 1)
    return (prediction == label).mean()
