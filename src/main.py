from datetime import datetime
from sklearn.utils import compute_class_weight
from tensorflow import keras
from src.data.load_data import train_X, train_y, validation_X, validation_y, test_X, test_y
from src.models.metrics import sps
from src.models.model import NextItemPredictor
from src.models.predict import predict_10
from src.models.train import train
import numpy as np

logdir = "../logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = NextItemPredictor()
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
)

class_weight = compute_class_weight('balanced', np.unique(train_y), train_y)
train(
    model,
    train_X=train_X,
    train_y=train_y,
    validation_X=validation_X,
    validation_y=validation_y,
    class_weight=class_weight,
    batch_size=32,
    epochs=1,
    callbacks=[tensorboard_callback],
    description='batch100-epochs30'
)
y_pred = predict_10(model, test_X)

sps_score = sps(test_y, y_pred)
f = open('../../logs/metrics_logs.txt', 'a')
f.write(str(datetime.now()) + '\n')
f.write('sps score: {} \n'.format(sps_score))
f.close()
