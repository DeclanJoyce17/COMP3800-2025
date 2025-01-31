import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import ast
import pickle

# Load existing model and preprocessing tools

with open('app/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('app/mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

with open('app/num_styles.pkl', 'rb') as file:
    num_styles = pickle.load(file)

with open('app/input_shape.pkl', 'rb') as file:
    input_shape = pickle.load(file)

# Load and preprocess new data
train_data = pd.read_csv('app/train.csv') # Old data
retrain_data = pd.read_csv('app/retrain.csv') # New data

def parse_styles(styles_str):
    try:
        return ast.literal_eval(styles_str)
    except Exception as e:
        print(f"Error parsing styles: {e}")
        return []

train_data['artist_styles'] = train_data['artist_styles'].apply(parse_styles)
retrain_data['artist_styles'] = retrain_data['artist_styles'].apply(parse_styles)

combined_data = pd.concat([train_data, retrain_data], ignore_index=True)

numeric_features = [
    'searched_artist_count', 'artist_description_read_count', 'artist_link_open_count',
    'style_description_read_count', 'style_image_view_styleguide_count', 'style_image_view_content_count',
    'style_list_open_count', 'purchased_item_review_count'
]

# Update the scaler with the combined data
scaler.fit(combined_data[numeric_features])
combined_data[numeric_features] = scaler.transform(combined_data[numeric_features])

# Update the MultiLabelBinarizer with the combined data
mlb.fit(combined_data['artist_styles'])
style_features = mlb.transform(combined_data['artist_styles'])
full_features_matrix = np.hstack([combined_data[numeric_features].values, style_features])

# Update the number of styles
num_styles = len(mlb.classes_)

# Update user and artist ID mappings
user_ids = combined_data['user_id'].astype('category').cat.codes.values
artist_ids = combined_data['artist_id'].astype('category').cat.codes.values
labels = np.ones(len(user_ids))

unique_user_ids = np.unique(user_ids)
unique_artist_ids = np.unique(artist_ids)
negative_samples = []

for user_id in unique_user_ids:
    positive_artists = artist_ids[user_ids == user_id]
    negative_artists = np.setdiff1d(unique_artist_ids, positive_artists)
    neg_samples_count = min(10, len(positive_artists), len(negative_artists))
    if neg_samples_count > 0:
        sampled_neg_artists = np.random.choice(negative_artists, size=neg_samples_count, replace=False)

        for neg_artist in sampled_neg_artists:
            negative_samples.append([user_id, neg_artist, 0])

positive_samples = np.column_stack((user_ids, artist_ids, labels))
negative_samples = np.array(negative_samples)

# Ensure there are enough negative samples
if len(negative_samples) < len(positive_samples):
    additional_negative_samples = []
    while len(additional_negative_samples) < (len(positive_samples) - len(negative_samples)):
        user_id = np.random.choice(unique_user_ids)
        negative_artists = np.setdiff1d(unique_artist_ids, artist_ids[user_ids == user_id])
        if len(negative_artists) > 0:
            neg_artist = np.random.choice(negative_artists)
            additional_negative_samples.append([user_id, neg_artist, 0])
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

train_full_features = np.vstack([full_features_matrix[train_artist_ids]])
test_full_features = np.vstack([full_features_matrix[test_artist_ids]])

# Redefine the model to update the embedding layers
num_users = len(np.unique(user_ids))
num_artists = len(np.unique(artist_ids))
num_numeric_features = len(numeric_features)

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

model = tf.keras.models.load_model('app/hybrid_recommender_model.keras', custom_objects={"HybridModel": HybridModel})

model = HybridModel(num_users, num_artists, num_numeric_features, num_styles)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Retrain the model
train_data = tf.data.Dataset.from_tensor_slices(((train_user_ids, train_artist_ids, train_full_features), train_labels))
train_data = train_data.shuffle(10000).batch(512)

val_data = tf.data.Dataset.from_tensor_slices(((test_user_ids, test_artist_ids, test_full_features), test_labels))
val_data = val_data.batch(512)

history = model.fit(train_data, epochs=10, validation_data=val_data)

# Save updated model and preprocessing tools
model.save('app/hybrid_recommender_model.keras')

with open('app/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('app/mlb.pkl', 'wb') as file:
    pickle.dump(mlb, file)

with open('app/num_styles.pkl', 'wb') as file:
    pickle.dump(num_styles, file)

with open('app/input_shape.pkl', 'wb') as file:
    pickle.dump(input_shape, file)
