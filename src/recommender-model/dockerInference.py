import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import ast
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

# Load the trained artifacts
with open('app/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('app/mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

with open('app/product_id_mapping.pkl', 'rb') as file:
    product_id_mapping = pickle.load(file)

with open('app/user_id_mapping.pkl', 'rb') as file:
    user_id_mapping = pickle.load(file)

with open('app/product_style_dict.pkl', 'rb') as file:
    product_style_dict = pickle.load(file)

with open('app/input_shape.pkl', 'rb') as file:
    input_shape = pickle.load(file)
    num_styles = input_shape[0] - 12  # 12 numeric features as defined in training

# Load the dataset
data = pd.read_csv('app/train.csv')

# Define numeric features (consistent with training)
numeric_features = [
    'product_view_count', 'product_description_read_count', 'searched_product_count', 
    'product_favourite_count', 'product_purchase_count', 'product_link_open_count', 
    'style_description_read_count', 'style_image_view_styleguide_count', 
    'style_image_view_content_count', 'product_added_to_cart_count', 
    'style_list_open_count', 'purchased_item_review_count'
]

# Normalize the numerical features
data[numeric_features] = scaler.transform(data[numeric_features])

# Get number of users and products
num_users = len(user_id_mapping)
num_products = len(product_id_mapping)

# Prepare the full feature matrix for all products
full_features_matrix = []
for product_id in product_id_mapping.keys():
    # For inference, assume no interaction (zero numeric features) for unseen user-product pairs
    num_feats = np.zeros(len(numeric_features), dtype=np.float32)
    style_feats = product_style_dict[product_id].astype(np.float32)
    full_feats = np.hstack([num_feats, style_feats])
    full_features_matrix.append(full_feats)
full_features_matrix = np.array(full_features_matrix)

# Load the model
class HybridModel(tf.keras.Model):
    def __init__(self, num_users, num_products, num_numeric_features, num_styles, embedding_size=64, **kwargs):
        super(HybridModel, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_products = num_products
        self.num_numeric_features = num_numeric_features
        self.num_styles = num_styles
        self.embedding_size = embedding_size
        
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal')
        self.product_embedding = tf.keras.layers.Embedding(num_products, embedding_size, embeddings_initializer='he_normal')
        self.numeric_features_layer = tf.keras.layers.Dense(embedding_size, activation='relu')
        self.style_features_layer = tf.keras.layers.Dense(embedding_size, activation='relu')
        self.concat_layer = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.hidden_1 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(64, activation='relu')
        self.hidden_3 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_id, product_id, full_features = inputs
        user_vec = self.user_embedding(user_id)
        product_vec = self.product_embedding(product_id)
        numeric_feature_vec = self.numeric_features_layer(full_features[:, :self.num_numeric_features])
        style_feature_vec = self.style_features_layer(full_features[:, self.num_numeric_features:])
        combined_features = self.concat_layer([user_vec, product_vec, numeric_feature_vec, style_feature_vec])
        x = self.dropout(combined_features)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        return self.dense_final(x)

    def get_config(self):
        config = super(HybridModel, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_products': self.num_products,
            'num_numeric_features': self.num_numeric_features,
            'num_styles': self.num_styles,
            'embedding_size': self.embedding_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_users=config['num_users'],
            num_products=config['num_products'],
            num_numeric_features=config['num_numeric_features'],
            num_styles=config['num_styles'],
            embedding_size=config['embedding_size']
        )

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.int32, name='input_1'),
        tf.TensorSpec(shape=(None,), dtype=tf.int32, name='input_2'),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32, name='input_3')
    ])
    def predict_signature(self, user_tensor, product_tensor, full_features_tensor):
        return self.call([user_tensor, product_tensor, full_features_tensor])

model = tf.keras.models.load_model('app/hybrid_recommender_model.keras', custom_objects={"HybridModel": HybridModel})

# Define the recommendation function
def get_top_n_recommendations(user_id, num_products, n=5):
    user_idx = user_id_mapping.get(user_id, -1)
    if user_idx == -1:
        raise ValueError(f"User ID {user_id} not found in the dataset.")

    all_product_ids = np.arange(num_products, dtype=np.int32)
    user_tensor = tf.convert_to_tensor(np.full(shape=(num_products,), fill_value=user_idx, dtype=np.int32))
    product_tensor = tf.convert_to_tensor(all_product_ids, dtype=tf.int32)
    full_features_tensor = tf.convert_to_tensor(full_features_matrix, dtype=tf.float32)

    predictions = model.predict_signature(user_tensor, product_tensor, full_features_tensor)
    top_indices = np.argsort(predictions.numpy().flatten())[-n:][::-1]
    return [list(product_id_mapping.keys())[i] for i in top_indices]

# Generate and store recommendations
recommendations_5_list = []
recommendations_list = []
for user_id in user_id_mapping.keys():
    try:
        top_5_recommendations = get_top_n_recommendations(user_id, num_products, n=5)
        top_recommendations = get_top_n_recommendations(user_id, num_products, n=50)
        recommendations_5_list.append({'User_ID': user_id, 'Top_5_Recommendations': top_5_recommendations})
        recommendations_list.append({'User_ID': user_id, 'Top_Recommendations': top_recommendations})
    except Exception as e:
        print(f"Error for User ID {user_id}: {e}")

# Convert to DataFrame
recommendations_5_df = pd.DataFrame(recommendations_5_list)
recommendations_df = pd.DataFrame(recommendations_list)

# Save to CSV
recommendations_5_df.to_csv('app/top_5_recommendations.csv', index=False)
recommendations_df.to_csv('app/top_n_recommendations.csv', index=False)
print("Recommendations saved to app/top_5_recommendations.csv")
print("Recommendations saved to app/top_n_recommendations.csv")

# Save to JSON
recommendations_5_json = recommendations_5_df.to_json(orient='records')
with open('app/user_5_recommendations.json', 'w') as json_file:
    json_file.write(recommendations_5_json)
print("Recommendations saved to app/user_5_recommendations.json")
recommendations_json = recommendations_df.to_json(orient='records')
with open('app/user_n_recommendations.json', 'w') as json_file:
    json_file.write(recommendations_json)
print("Recommendations saved to app/user_n_recommendations.json")