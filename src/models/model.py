from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras import initializers
from tensorflow_core.python.keras.models import Sequential

model = Sequential()
model.add(Embedding(input_dim=3953, output_dim=400, embeddings_initializer=initializers.Zeros()))
model.add(LSTM(64))
model.add(Dense(3953, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy')