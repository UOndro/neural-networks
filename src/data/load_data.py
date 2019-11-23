import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow import keras

SEQUENCE_LENGTH = 50


def get_labeled_features(df):
    X = []
    y = []
    for _, group in df.groupby(['user_id']):
        # DATASET should be sorted !!!!!
        sequence = group[group['rating'] > 2]['movie_id'].tolist()
        if not sequence:
            continue
        chunks = [sequence[x:x + SEQUENCE_LENGTH] for x in range(0, len(sequence), SEQUENCE_LENGTH)]
        for chunk in chunks:
            if len(chunk) < SEQUENCE_LENGTH / 2:
                y.append(chunk.pop())
                X.append(chunk)
    X = keras.preprocessing.sequence.pad_sequences(X, padding='post', maxlen=200)
    return X, np.array(y)


df_ratings = pd.read_csv('../data/raw/ratings.dat', sep='::', encoding="ISO-8859-1",
                         names=['user_id', 'movie_id', 'rating', 'timestamp'])
df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s')
df_ratings.sort_values(by=['timestamp'], inplace=True)
train, validate, test = np.split(df_ratings, [int(.6 * len(df_ratings)), int(.8 * len(df_ratings))])

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df_ratings['movie_id'].unique())
train['movie_id'] = label_encoder.transform(train['movie_id'])
validate['movie_id'] = label_encoder.transform(validate['movie_id'])
test['movie_id'] = label_encoder.transform(test['movie_id'])

train_X, train_y = get_labeled_features(train)
validation_X, validation_y = get_labeled_features(validate)
test_X, test_y = get_labeled_features(test)
