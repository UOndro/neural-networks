import numpy as np


def predict(model, test_X):
    return np.apply_along_axis(lambda seq: model.predict_classes(seq)[-1], 1, test_X)


def predict_10(model, test_X):
    predictions = model.predict(test_X)
    result = []
    for prediction in predictions:
        top_10 = np.argpartition(prediction, -10)[-10:]
        result.append(top_10[np.argsort(prediction[top_10])])
    return result

