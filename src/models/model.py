import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras import initializers
from tensorflow_core.python.keras.models import Sequential

from src.data.load_data import train_y, df_ratings


class NextItemPredictor(Sequential):
    def __init__(self):
        super(NextItemPredictor, self).__init__()
        self.add(Embedding(input_dim=len(np.unique(df_ratings['movie_id'])), output_dim=400, embeddings_initializer=initializers.Zeros()))
        self.add(LSTM(1024))
        self.add(Dense(len(np.unique(df_ratings['movie_id'])), activation='softmax'))

    def compile(self):
        super(NextItemPredictor, self).compile(optimizer='adam', loss='sparse_categorical_crossentropy')
