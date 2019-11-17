from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras import initializers
from tensorflow_core.python.keras.models import Sequential


class NextItemPredictor(Sequential):
    def __init__(self):
        super(NextItemPredictor, self).__init__()
        self.add(Embedding(input_dim=3953, output_dim=400, embeddings_initializer=initializers.Zeros()))
        self.add(LSTM(64))
        self.add(Dense(3953, activation='softmax'))

    def compile(self):
        super(NextItemPredictor, self).compile(optimizer='adam', loss='sparse_categorical_crossentropy')
