import logging
import os
from datetime import datetime
from sklearn.utils import compute_class_weight
from tensorflow import keras
from src.data.load_data import train_X, train_y, validation_X, validation_y, test_X, test_y
from src.models.metrics import sps, item_coverage
from src.models.model import NextItemPredictor
from src.models.predict import predict_10
from src.models.train import train
import numpy as np

# %%
logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
os.mkdir(logdir)
logging.basicConfig(filename='{}/metric_logs'.format(logdir), level=logging.DEBUG)
# %%
batch_size = 64
epochs = 32
lstm_size = 256
emb_size = 200
logging.info('Batch {}. Epochs {} LSTM {}'.format(batch_size, epochs, lstm_size, emb_size))
# %%
model = NextItemPredictor(lstm_size, emb_size)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
# %%
class_weight = compute_class_weight('balanced', np.unique(train_y), train_y)
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
train(
    model,
    train_X=train_X,
    train_y=train_y,
    validation_X=validation_X,
    validation_y=validation_y,
    class_weight=class_weight,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[es, tensorboard_callback],
    description='batch{}-epochs{}-lstm{}'.format(batch_size, epochs, lstm_size)
)
# %%
y_pred = predict_10(model, test_X)
# %%
sps(test_y, y_pred)
# %%
item_coverage(test_y, y_pred)


