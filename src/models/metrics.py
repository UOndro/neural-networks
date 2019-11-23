import numpy as np
from sklearn.metrics import recall_score, precision_score


def sps(y_true, y_pred):
    correct = 0.0
    for i, y in enumerate(y_true):
        if y in y_pred[i]:
            correct += 1
    return correct / len(y_true)


def recall(y_true, y_pred):
    recall_score(y_true, y_pred, average='weighted')


def avg_precision(y_true, y_pred):
    return np.average(precision_score(y_true, y_pred, average=None))
