import os
from time import time


def train(model, train_X, train_y, validation_X, validation_y, class_weight, epochs=30, batch_size=10, callbacks=None,
          description=''):
    model.fit(
        x=train_X,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_X, validation_y),
        callbacks=callbacks,
        # class_weight=class_weight
    )
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../models/model_{}_{}'.format(int(time()), description)
    )
    model.save(data_path, save_format='tf')
