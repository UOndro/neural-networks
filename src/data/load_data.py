import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
import os

SEQUENCE_LENGTH = 51
DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../data/raw/'
)


def load_movies():
    df = pd.read_csv(os.path.join(DATA_PATH, 'movies.dat'), sep='::', encoding="ISO-8859-1",
                     names=['movie_id', 'title', 'genres'])
    df['genres'] = df['genres'].str.split('|')
    return df


def load_ratings():
    df = pd.read_csv(os.path.join(DATA_PATH, 'ratings.dat'), sep='::', encoding="ISO-8859-1",
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by=['timestamp'], inplace=True)
    return df


def make_movie_id_encoder(movie_ids):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(movie_ids)
    return encoder


def make_genres_encoder(genres):
    encoder = MultiLabelBinarizer()
    encoder.fit(genres)
    return encoder


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
    X = keras.preprocessing.sequence.pad_sequences(X, padding='pre', maxlen=SEQUENCE_LENGTH - 1)
    return X, np.array(y)


df_movies = load_movies()
df_ratings = load_ratings()

movie_id_encoder = make_movie_id_encoder(df_movies['movie_id'].unique())
df_ratings['movie_id'] = movie_id_encoder.transform(df_ratings['movie_id'])
df_movies['movie_id'] = movie_id_encoder.transform(df_movies['movie_id'])

train, validate, test = np.split(df_ratings, [int(.6 * len(df_ratings)), int(.8 * len(df_ratings))])

train_X, train_y = get_labeled_features(train)
validation_X, validation_y = get_labeled_features(validate)
test_X, test_y = get_labeled_features(test)

genres_encoder = make_genres_encoder(df_movies['genres'])
genres_encoding = genres_encoder.transform(df_movies['genres'])
