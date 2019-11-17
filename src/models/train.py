from time import time


def fit(model, train_X, train_y, validation_X, validation_y, epochs=30, batch_size=10):
    model.fit(
        x=train_X,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_X, validation_y))
    model.save(f'../models/model_{int(time())}.h5')
