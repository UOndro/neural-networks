import logging

import numpy as np
from sklearn.metrics import recall_score, precision_score

logger = logging.getLogger(__name__)


def sps(y_true, y_pred):
    y = np.column_stack((y_true, y_pred))
    hits = np.apply_along_axis(lambda y_: y_[0] in y_[1:], 1, y)
    result = sum(hits) / len(y_true)
    logger.info('SPS metric {}'.format(result))
    return result


def item_coverage(y_true, y_pred):
    y = np.column_stack((y_true, y_pred))
    hits = np.apply_along_axis(lambda y_: y_[0] in y_[1:], 1, y)
    result = len(np.unique(y_true[hits])) / len(np.unique(y_true))
    logger.info('Item covarage {}'.format(result))
    return result


def recall(y_true, y_pred):
    recall_score(y_true, y_pred, average='weighted')


def avg_precision(y_true, y_pred):
    return np.average(precision_score(y_true, y_pred, average=None))
