import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import ast

def save_input_shape(input_shape, path):
    with open(path, 'wb') as file:
        pickle.dump(input_shape, file)

def load_scaler(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def load_mlb(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_scaler(scaler, path):
    with open(path, 'wb') as file:
        pickle.dump(scaler, file)

def save_mlb(mlb, path):
    with open(path, 'wb') as file:
        pickle.dump(mlb, file)

def save_num_styles(num_styles, path):
    with open(path, 'wb') as file:
        pickle.dump(num_styles, file)

def load_num_styles(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def load_input_shape(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def parse_styles(styles_str):
    try:
        return ast.literal_eval(styles_str)
    except Exception as e:
        print(f"Error parsing styles: {e}")
        return []

def preprocess_data(data, scaler, mlb, numeric_features):
    data['artist_styles'] = data['artist_styles'].apply(parse_styles)
    data[numeric_features] = scaler.transform(data[numeric_features])
    style_features = mlb.transform(data['artist_styles'])
    combined_features = np.hstack([data[numeric_features].values, style_features])
    return combined_features, style_features.shape[1]

def create_mappings(data):
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(data['user_id'].unique())}
    artist_id_mapping = {artist_id: idx for idx, artist_id in enumerate(data['artist_id'].unique())}
    return user_id_mapping, artist_id_mapping

def create_model(num_users, num_artists, num_numeric_features, num_styles, embedding_size=64):
    class HybridModel(tf.keras.Model):
        def __init__(self, num_users, num_artists, num_numeric_features, num_styles, embedding_size):
            super(HybridModel, self).__init__()
            self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal')
            self.artist_embedding = tf.keras.layers.Embedding(num_artists, embedding_size, embeddings_initializer='he_normal')
            self.numeric_features_layer = tf.keras.layers.Dense(embedding_size, activation='relu')
            self.style_features_layer = tf.keras.layers.Dense(embedding_size, activation='relu')
            self.concat_layer = tf.keras.layers.Concatenate()
            self.dropout = tf.keras.layers.Dropout(0.4)
            self.hidden_1 = tf.keras.layers.Dense(128, activation='relu')
            self.hidden_2 = tf.keras.layers.Dense(64, activation='relu')
            self.hidden_3 = tf.keras.layers.Dense(32, activation='relu')
            self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid')

        def call(self, inputs):
            user_id, artist_id, full_features = inputs
            user_vec = self.user_embedding(user_id)
            artist_vec = self.artist_embedding(artist_id)
            numeric_feature_vec = self.numeric_features_layer(full_features[:, :num_numeric_features])
            style_feature_vec = self.style_features_layer(full_features[:, num_numeric_features:])
            combined_features = self.concat_layer([user_vec, artist_vec, numeric_feature_vec, style_feature_vec])
            x = self.dropout(combined_features)
            x = self.hidden_1(x)
            x = self.hidden_2(x)
            x = self.hidden_3(x)
            return self.dense_final(x)

    model = HybridModel(num_users, num_artists, num_numeric_features, num_styles, embedding_size)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
