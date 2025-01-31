import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import ast
import pickle

# Define numeric features
numeric_features = [
    'searched_artist_count', 'artist_description_read_count', 'artist_link_open_count',
    'style_description_read_count', 'style_image_view_styleguide_count', 'style_image_view_content_count',
    'style_list_open_count', 'purchased_item_review_count'
]

# Load data
data = pd.read_csv('app/train.csv')

# Convert artist_styles strings to lists
def parse_styles(styles_str):
    try:
        return ast.literal_eval(styles_str)
    except Exception as e:
        print(f"Error parsing styles: {e}")
        return []

data['artist_styles'] = data['artist_styles'].apply(parse_styles)

# Normalize the numerical features
scaler = MinMaxScaler().fit(data[numeric_features])
data[numeric_features] = scaler.transform(data[numeric_features])

# Encode artist styles using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
style_features = mlb.fit_transform(data['artist_styles'])
num_styles = len(mlb.classes_)

# Save the number of styles
with open('app/num_styles.pkl', 'wb') as file:
    pickle.dump(num_styles, file)

# Prepare collaborative filtering data
user_ids = data['user_id'].values
artist_ids = data['artist_id'].values

# Get unique user and artist IDs
unique_user_ids = np.unique(user_ids)
unique_artist_ids = np.unique(artist_ids)

# Get user and artist mappings
user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
artist_id_mapping = {artist_id: idx for idx, artist_id in enumerate(unique_artist_ids)}

num_users = len(unique_user_ids)
num_artists = len(unique_artist_ids)

# Map user_ids and artist_ids to indices
mapped_user_ids = np.array([user_id_mapping[uid] for uid in user_ids])
mapped_artist_ids = np.array([artist_id_mapping[aid] for aid in artist_ids])

# Concatenate numerical and style features
numeric_features_matrix = data[numeric_features].values
full_features_matrix = np.hstack([numeric_features_matrix, style_features])

# Create positive labels (1s)
labels = np.ones(len(mapped_user_ids))

# Create negative samples
negative_samples = []

for user_id in unique_user_ids:
    positive_artists = artist_ids[user_ids == user_id]
    negative_artists = np.setdiff1d(unique_artist_ids, positive_artists)
    neg_samples_count = min(10, len(positive_artists), len(negative_artists))
    if neg_samples_count > 0:
        sampled_neg_artists = np.random.choice(negative_artists, size=neg_samples_count, replace=False)
        for neg_artist in sampled_neg_artists:
            negative_samples.append([user_id_mapping[user_id], artist_id_mapping[neg_artist], 0])

# Combine positive and negative samples
positive_samples = np.column_stack((mapped_user_ids, mapped_artist_ids, labels))
negative_samples = np.array(negative_samples)

if len(negative_samples) < len(positive_samples):
    additional_negative_samples = []
    while len(additional_negative_samples) < (len(positive_samples) - len(negative_samples)):
        user_id = np.random.choice(unique_user_ids)
        negative_artists = np.setdiff1d(unique_artist_ids, artist_ids[user_ids == user_id])
        if len(negative_artists) > 0:
            neg_artist = np.random.choice(negative_artists)
            additional_negative_samples.append([user_id_mapping[user_id], artist_id_mapping[neg_artist], 0])
    negative_samples = np.vstack((negative_samples, additional_negative_samples))

combined_samples = np.vstack((positive_samples, negative_samples))
np.random.shuffle(combined_samples)

train_samples, test_samples = train_test_split(combined_samples, test_size=0.2, random_state=42)

train_user_ids = train_samples[:, 0].astype(np.int32)
train_artist_ids = train_samples[:, 1].astype(np.int32)
train_labels = train_samples[:, 2].astype(np.float32)

test_user_ids = test_samples[:, 0].astype(np.int32)
test_artist_ids = test_samples[:, 1].astype(np.int32)
test_labels = test_samples[:, 2].astype(np.float32)

train_full_features = full_features_matrix[train_artist_ids]
test_full_features = full_features_matrix[test_artist_ids]

class HybridModel(tf.keras.Model):
    def __init__(self, num_users, num_artists, num_numeric_features, num_styles, embedding_size=64, **kwargs):
        super(HybridModel, self).__init__(**kwargs)  # Pass kwargs to parent class to handle Keras' internal params
        self.num_users = num_users
        self.num_artists = num_artists
        self.num_numeric_features = num_numeric_features
        self.num_styles = num_styles
        self.embedding_size = embedding_size
        
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
        numeric_feature_vec = self.numeric_features_layer(full_features[:, :self.num_numeric_features])
        style_feature_vec = self.style_features_layer(full_features[:, self.num_numeric_features:])
        combined_features = self.concat_layer([user_vec, artist_vec, numeric_feature_vec, style_feature_vec])
        x = self.dropout(combined_features)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        return self.dense_final(x)

    def get_config(self):
        config = super(HybridModel, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_artists': self.num_artists,
            'num_numeric_features': self.num_numeric_features,
            'num_styles': self.num_styles,
            'embedding_size': self.embedding_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_users=config['num_users'],
            num_artists=config['num_artists'],
            num_numeric_features=config['num_numeric_features'],
            num_styles=config['num_styles'],
            embedding_size=config['embedding_size']
        )

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.int32, name='input_1'),
        tf.TensorSpec(shape=(None,), dtype=tf.int32, name='input_2'),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32, name='input_3')
    ])
    def predict_signature(self, user_tensor, artist_tensor, full_features_tensor):
        return self.call([user_tensor, artist_tensor, full_features_tensor])

model = HybridModel(num_users, num_artists, len(numeric_features), num_styles)
model.compile(optimizer='adam', loss='binary_crossentropy')

train_data = tf.data.Dataset.from_tensor_slices(((train_user_ids, train_artist_ids, train_full_features), train_labels))
train_data = train_data.shuffle(10000).batch(512)

val_data = tf.data.Dataset.from_tensor_slices(((test_user_ids, test_artist_ids, test_full_features), test_labels))
val_data = val_data.batch(512)

history = model.fit(train_data, epochs=150, validation_data=val_data)

model.save('app/hybrid_recommender_model.keras')

with open('app/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('app/mlb.pkl', 'wb') as file:
    pickle.dump(mlb, file)

input_shape = (full_features_matrix.shape[1],)
with open('app/input_shape.pkl', 'wb') as file:
    pickle.dump(input_shape, file)
