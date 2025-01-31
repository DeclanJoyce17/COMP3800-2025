import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import ast
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

# Load the trained model with the custom object
with open('app/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('app/mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

# Load num_styles from the pickled file
with open('app/num_styles.pkl', 'rb') as file:
    num_styles = pickle.load(file)

# Load the dataset
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
numeric_features = [
    'searched_artist_count', 'artist_description_read_count', 'artist_link_open_count',
    'style_description_read_count', 'style_image_view_styleguide_count', 'style_image_view_content_count',
    'style_list_open_count', 'purchased_item_review_count'
]
data[numeric_features] = scaler.transform(data[numeric_features])

# Encode artist styles
style_features = mlb.transform(data['artist_styles'])

# Prepare full feature matrix
full_features_matrix = np.hstack([data[numeric_features].values, style_features])

# Get user and artist mappings
user_id_mapping = {user_id: idx for idx, user_id in enumerate(data['user_id'].unique())}
artist_id_mapping = {artist_id: idx for idx, artist_id in enumerate(data['artist_id'].unique())}
num_users = len(user_id_mapping)
num_artists = len(artist_id_mapping)

# Ensure the full_features_matrix has the correct number of rows for each artist
full_features_matrix = np.tile(full_features_matrix, (num_artists, 1))

# Load the model
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

# Define the recommendation function
def get_top_n_recommendations(user_id, num_artists, n=5):
    user_idx = user_id_mapping.get(user_id, -1)
    if user_idx == -1:
        raise ValueError(f"User ID {user_id} not found in the dataset.")

    all_artist_ids = np.arange(num_artists, dtype=np.int32)
    user_tensor = tf.convert_to_tensor(np.full(shape=(num_artists,), fill_value=user_idx, dtype=np.int32))
    artist_tensor = tf.convert_to_tensor(all_artist_ids, dtype=tf.int32)    

    # Select the rows corresponding to the user
    full_features_tensor = tf.convert_to_tensor(full_features_matrix[:num_artists], dtype=tf.float32)

    predictions = model.predict_signature(user_tensor, artist_tensor, full_features_tensor)
    top_indices = np.argsort(predictions.numpy().flatten())[-n:][::-1]
    return [list(artist_id_mapping.keys())[i] for i in top_indices]

# Create DataFrame to store recommendations
recommendations_list = []

# Generate recommendations for each user
for user_id in user_id_mapping.keys():
    try:
        top_recommendations = get_top_n_recommendations(user_id, num_artists)
        recommendations_list.append({'User_ID': user_id, 'Top_Recommendations': top_recommendations})
    except Exception as e:
        print(f"Error for User ID {user_id}: {e}")

# Convert recommendations list to DataFrame
recommendations_df = pd.DataFrame(recommendations_list)

# Save recommendations to CSV file
recommendations_df.to_csv('app/top_5_recommendations.csv', index=False)
print("Recommendations saved to app/top_5_recommendations.csv")

# Convert DataFrame to JSON
recommendations_json = recommendations_df.to_json(orient='records')

# Write JSON to a file
with open('app/user_recommendations.json', 'w') as json_file:
    json_file.write(recommendations_json)

print("Recommendations saved to app/user_recommendations.json")
