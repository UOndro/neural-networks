import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Layer, concatenate
from tensorflow_core.python.keras.models import Model

from src.data.load_data import df_movies


class NextItemPredictor(Model):
    def __init__(self, size_lstm=124, emb_out=400):
        super(NextItemPredictor, self).__init__()
        self.emb = Embedding(input_dim=len(df_movies['movie_id']), output_dim=emb_out, mask_zero=True)
        self.lstm = LSTM(size_lstm, activation='sigmoid')
        self.out = Dense(len(df_movies['movie_id']), activation='softmax')

    def call(self, x):
        emb = self.emb(x[0])
        con = concatenate([emb, tf.cast(x[1], dtype=tf.float32)])
        lstm = self.lstm(con)
        return self.out(lstm)
