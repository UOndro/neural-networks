import logging
import os
from datetime import datetime

import numpy as np

from src.data.load_data import test_X, test_y, df_ratings
from src.models.metrics import sps, item_coverage
from src.models.model import MostPopular

# %%
logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(logdir)
logging.basicConfig(filename='{}/metric_logs'.format(logdir), level=logging.DEBUG)
# %%
train, _, test = np.split(df_ratings, [int(.6 * len(df_ratings)), int(.8 * len(df_ratings))])
# %%
model = MostPopular()
# %%
model.train(train)
# %%
y_pred = model.predict(test_X[0], first_n=10)
# %%
sps(test_y, y_pred)
# %%
item_coverage(test_y, y_pred)
