import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import ast
import pickle

# Define numeric features specific to user-product interactions
numeric_features = [
    'product_view_count', 'product_description_read_count', 'searched_product_count', 
    'product_favourite_count', 'product_purchase_count', 'product_link_open_count', 
    'style_description_read_count', 'style_image_view_styleguide_count', 
    'style_image_view_content_count', 'product_added_to_cart_count', 
    'style_list_open_count', 'purchased_item_review_count'
]

### Step 1: Load and Process Product Data
# Load products.csv which contains all product information
products_df = pd.read_csv('app/products.csv')
# Convert string representations of lists in 'styles' column to actual lists
products_df['styles'] = products_df['styles'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Encode style features for all products using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
style_features_all = mlb.fit_transform(products_df['styles'])
num_styles = len(mlb.classes_)

# Create a dictionary mapping product IDs to their style features
product_style_dict = dict(zip(products_df['id'], style_features_all))

# Get all unique product IDs and count them
unique_product_ids = products_df['id'].unique()
num_products = len(unique_product_ids)

### Step 2: Load and Process Interaction Data
# Load train.csv which contains user-product interaction data
train_df = pd.read_csv('app/train.csv')

# Normalize interaction-specific numerical features
scaler = MinMaxScaler().fit(train_df[numeric_features])
train_df[numeric_features] = scaler.transform(train_df[numeric_features])

# Get unique user IDs and count them
unique_user_ids = train_df['user_id'].unique()
num_users = len(unique_user_ids)

### Step 3: Map IDs to Indices
# Create mappings from user and product IDs to indices
user_id_mapping = {uid: idx for idx, uid in enumerate(unique_user_ids)}
product_id_mapping = {pid: idx for idx, pid in enumerate(unique_product_ids)}

# Map user and product IDs in train_df to indices for positive samples
mapped_user_ids = np.array([user_id_mapping[uid] for uid in train_df['user_id']])
mapped_product_ids = np.array([product_id_mapping[pid] for pid in train_df['product_id']])
labels = np.ones(len(mapped_user_ids))  # Positive samples have label 1

# Stack user IDs, product IDs, and labels for positive samples
positive_samples = np.column_stack((mapped_user_ids, mapped_product_ids, labels))

### Step 4: Generate Negative Samples
# Generate negative samples (products a user hasn’t interacted with)
negative_samples = []
for user_id in unique_user_ids:
    # Products the user has interacted with
    positive_products = train_df[train_df['user_id'] == user_id]['product_id'].values
    # Products the user hasn’t interacted with
    negative_products = np.setdiff1d(unique_product_ids, positive_products)
    # Limit negative samples to balance with positive samples (up to 10 per user)
    neg_samples_count = min(10, len(positive_products), len(negative_products))
    if neg_samples_count > 0:
        sampled_neg_products = np.random.choice(negative_products, size=neg_samples_count, replace=False)
        for neg_product in sampled_neg_products:
            neg_product_idx = product_id_mapping[neg_product]
            negative_samples.append([user_id_mapping[user_id], neg_product_idx, 0])  # Label 0 for negative

negative_samples = np.array(negative_samples)

### Step 5: Combine and Split Samples
# Combine positive and negative samples, shuffle, and split into train/test sets
combined_samples = np.vstack((positive_samples, negative_samples))
np.random.shuffle(combined_samples)
train_samples, test_samples = train_test_split(combined_samples, test_size=0.2, random_state=42)

### Step 6: Prepare Features
# Features for positive samples (interaction + style features)
positive_features = []
for i in range(len(train_df)):
    num_feats = train_df.iloc[i][numeric_features].values.astype(np.float32)  # Interaction-specific features
    style_feats = product_style_dict[train_df.iloc[i]['product_id']].astype(np.float32)  # Product-specific style features
    full_feats = np.hstack([num_feats, style_feats])
    positive_features.append(full_feats)
positive_features = np.array(positive_features)

# Features for negative samples (zero interaction + style features)
negative_features = []
for user_idx, product_idx, label in negative_samples:
    product_id = unique_product_ids[product_idx]
    num_feats = np.zeros(len(numeric_features),dtype=np.float32)  # No interactions, so set to zero
    style_feats = product_style_dict[product_id].astype(np.float32)
    full_feats = np.hstack([num_feats, style_feats])
    negative_features.append(full_feats)
negative_features = np.array(negative_features)

# Combine and split features to align with samples
all_features = np.vstack((positive_features, negative_features))
train_features, test_features = train_test_split(all_features, test_size=0.2, random_state=42, shuffle=True)

# Extract user IDs, product IDs, and labels for training and validation
train_user_ids = train_samples[:, 0].astype(np.int32)
train_product_ids = train_samples[:, 1].astype(np.int32)
train_labels = train_samples[:, 2].astype(np.float32)

test_user_ids = test_samples[:, 0].astype(np.int32)
test_product_ids = test_samples[:, 1].astype(np.int32)
test_labels = test_samples[:, 2].astype(np.float32)

### Step 7: Define the Hybrid Model
class HybridModel(tf.keras.Model):
    def __init__(self, num_users, num_products, num_numeric_features, num_styles, embedding_size=64, **kwargs):
        super(HybridModel, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_products = num_products
        self.num_numeric_features = num_numeric_features
        self.num_styles = num_styles
        self.embedding_size = embedding_size
        
        # Layers for embeddings and features
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

### Step 8: Train the Model
# Instantiate and compile the model
model = HybridModel(num_users, num_products, len(numeric_features), num_styles)
model.compile(optimizer='adam', loss='binary_crossentropy')

# train_features = train_features.astype(np.float32)
# print(train_features)
# Debug shapes and types
print("train_user_ids shape:", train_user_ids.shape, "dtype:", train_user_ids.dtype)
print("train_product_ids shape:", train_product_ids.shape, "dtype:", train_product_ids.dtype)
print("train_features shape:", train_features.shape, "dtype:", train_features.dtype)
print("train_labels shape:", train_labels.shape, "dtype:", train_labels.dtype)

# Prepare TensorFlow datasets
train_data = tf.data.Dataset.from_tensor_slices(((train_user_ids, train_product_ids, train_features), train_labels))
train_data = train_data.shuffle(10000).batch(512)

val_data = tf.data.Dataset.from_tensor_slices(((test_user_ids, test_product_ids, test_features), test_labels))
val_data = val_data.batch(512)

# Train the model
history = model.fit(train_data, epochs=150, validation_data=val_data)

### Step 9: Save the Model and Artifacts
# Save the trained model
model.save('app/hybrid_recommender_model.keras')

# Save preprocessing objects and mappings
with open('app/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('app/mlb.pkl', 'wb') as file:
    pickle.dump(mlb, file)

with open('app/product_id_mapping.pkl', 'wb') as file:
    pickle.dump(product_id_mapping, file)

with open('app/user_id_mapping.pkl', 'wb') as file:
    pickle.dump(user_id_mapping, file)

with open('app/product_style_dict.pkl', 'wb') as file:
    pickle.dump(product_style_dict, file)

# Save input shape for feature construction during prediction
input_shape = (len(numeric_features) + num_styles,)
with open('app/input_shape.pkl', 'wb') as file:
    pickle.dump(input_shape, file)