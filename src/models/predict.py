import numpy as np
from sklearn.metrics import precision_score


def predict(model, test_X):
    return np.apply_along_axis(lambda seq: model.predict_classes(seq)[-1], 1, test_X)


def avg_precision(y_true, y_pred):
    return np.average(precision_score(y_true, y_pred, average=None))
