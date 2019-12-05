import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense, Embedding, concatenate
from tensorflow_core.python.keras.models import Model

from src.data.load_data import df_movies


class NextItemPredictor(Model):
    def __init__(self, size_lstm=124, emb_out=400):
        super(NextItemPredictor, self).__init__()
        self.emb = Embedding(input_dim=len(df_movies['movie_id']), output_dim=emb_out, mask_zero=True)
        self.lstm = LSTM(size_lstm, activation='sigmoid',
                         kernel_regularizer=keras.regularizers.l2(0.01),
                         activity_regularizer=keras.regularizers.l1(0.01)
                         )
        self.out = Dense(len(df_movies['movie_id']), activation='softmax',
                         kernel_regularizer=keras.regularizers.l2(0.01),
                         activity_regularizer=keras.regularizers.l1(0.01))

    def call(self, x):
        emb = self.emb(x[0])
        con = concatenate([emb, tf.cast(x[1], dtype=tf.float32)])
        lstm = self.lstm(con)
        return self.out(lstm)


class MostPopular:
    def __init__(self):
        self.most_popular = None

    def train(self, df: pd.DataFrame):
        self.most_popular = df['movie_id'].value_counts(sort=True).index.tolist()

    def predict(self, X, first_n=None):
        first_n = first_n or len(self.most_popular)
        return np.asarray([self.most_popular[:first_n] for _ in X])
