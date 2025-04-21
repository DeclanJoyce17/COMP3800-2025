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

# Define the HybridModel class (must match dockerTrain.py)
class HybridModel(tf.keras.Model):
    def __init__(self, num_users, num_products, num_numeric_features, num_styles, embedding_size=64, **kwargs):
        super(HybridModel, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_products = num_products
        self.num_numeric_features = num_numeric_features
        self.num_styles = num_styles
        self.embedding_size = embedding_size
        
        # Layers for embeddings and features
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', name='user_embedding')
        self.product_embedding = tf.keras.layers.Embedding(num_products, embedding_size, embeddings_initializer='he_normal', name='product_embedding')
        self.numeric_features_layer = tf.keras.layers.Dense(embedding_size, activation='relu', name='numeric_features_layer')
        self.style_features_layer = tf.keras.layers.Dense(embedding_size, activation='relu', name='style_features_layer')
        self.concat_layer = tf.keras.layers.Concatenate(name='concat_layer')
        self.dropout = tf.keras.layers.Dropout(0.4, name='dropout')
        self.hidden_1 = tf.keras.layers.Dense(128, activation='relu', name='hidden_1')
        self.hidden_2 = tf.keras.layers.Dense(64, activation='relu', name='hidden_2')
        self.hidden_3 = tf.keras.layers.Dense(32, activation='relu', name='hidden_3')
        self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_final')

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

# Step 1: Load existing artifacts
with open('app/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('app/mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)
with open('app/user_id_mapping.pkl', 'rb') as file:
    old_user_id_mapping = pickle.load(file)
with open('app/product_id_mapping.pkl', 'rb') as file:
    old_product_id_mapping = pickle.load(file)
with open('app/product_style_dict.pkl', 'rb') as file:
    old_product_style_dict = pickle.load(file)
with open('app/input_shape.pkl', 'rb') as file:
    input_shape = pickle.load(file)

# Load the existing model
old_model = tf.keras.models.load_model('app/hybrid_recommender_model.keras', custom_objects={'HybridModel': HybridModel})

# Step 2: Load and process updated product data
products_df = pd.read_csv('app/products.csv')
products_df['styles'] = products_df['styles'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
style_features_all = mlb.transform(products_df['styles'])  # Use existing MLB, new styles ignored
product_style_dict = dict(zip(products_df['id'], style_features_all))

# Step 3: Load and combine interaction data
train_df = pd.read_csv('app/train.csv')
retrain_df = pd.read_csv('app/retrain.csv')
combined_df = pd.concat([train_df, retrain_df], ignore_index=True)

# Normalize numerical features using existing scaler
combined_df[numeric_features] = scaler.transform(combined_df[numeric_features])

# Step 4: Update mappings
unique_user_ids = combined_df['user_id'].unique()
unique_product_ids = products_df['id'].unique()
user_id_mapping = {uid: idx for idx, uid in enumerate(unique_user_ids)}
product_id_mapping = {pid: idx for idx, pid in enumerate(unique_product_ids)}

# Step 5: Prepare positive samples
mapped_user_ids = np.array([user_id_mapping[uid] for uid in combined_df['user_id']])
mapped_product_ids = np.array([product_id_mapping[pid] for pid in combined_df['product_id']])
labels = np.ones(len(mapped_user_ids))
positive_samples = np.column_stack((mapped_user_ids, mapped_product_ids, labels))

# Step 6: Generate negative samples
negative_samples = []
for user_id in unique_user_ids:
    positive_products = combined_df[combined_df['user_id'] == user_id]['product_id'].values
    negative_products = np.setdiff1d(unique_product_ids, positive_products)
    neg_samples_count = min(10, len(positive_products), len(negative_products))
    if neg_samples_count > 0:
        sampled_neg_products = np.random.choice(negative_products, size=neg_samples_count, replace=False)
        for neg_product in sampled_neg_products:
            neg_product_idx = product_id_mapping[neg_product]
            negative_samples.append([user_id_mapping[user_id], neg_product_idx, 0])

negative_samples = np.array(negative_samples)

# Step 7: Combine and split samples
combined_samples = np.vstack((positive_samples, negative_samples))
np.random.shuffle(combined_samples)
train_samples, test_samples = train_test_split(combined_samples, test_size=0.2, random_state=42)

# Step 8: Prepare features
positive_features = []
for i in range(len(combined_df)):
    num_feats = combined_df.iloc[i][numeric_features].values.astype(np.float32)
    style_feats = product_style_dict[combined_df.iloc[i]['product_id']].astype(np.float32)
    full_feats = np.hstack([num_feats, style_feats])
    positive_features.append(full_feats)
positive_features = np.array(positive_features)

negative_features = []
for user_idx, product_idx, label in negative_samples:
    product_id = unique_product_ids[product_idx]
    num_feats = np.zeros(len(numeric_features), dtype=np.float32)
    style_feats = product_style_dict[product_id].astype(np.float32)
    full_feats = np.hstack([num_feats, style_feats])
    negative_features.append(full_feats)
negative_features = np.array(negative_features)

all_features = np.vstack((positive_features, negative_features))
train_features, test_features = train_test_split(all_features, test_size=0.2, random_state=42, shuffle=True)

# Step 9: Extract inputs for training
train_user_ids = train_samples[:, 0].astype(np.int32)
train_product_ids = train_samples[:, 1].astype(np.int32)
train_labels = train_samples[:, 2].astype(np.float32)
test_user_ids = test_samples[:, 0].astype(np.int32)
test_product_ids = test_samples[:, 1].astype(np.int32)
test_labels = test_samples[:, 2].astype(np.float32)

# Step 10: Update the model
num_users = len(unique_user_ids)
num_products = len(unique_product_ids)
num_numeric_features = len(numeric_features)
num_styles = len(mlb.classes_)
model = HybridModel(num_users, num_products, num_numeric_features, num_styles)

# Build the model by calling it with a small batch of training data
small_batch_size = 1
small_user_ids = train_user_ids[:small_batch_size].astype(np.int32)
small_product_ids = train_product_ids[:small_batch_size].astype(np.int32)
small_features = train_features[:small_batch_size].astype(np.float32)

# Pass the small batch through the model to initialize weights
_ = model([small_user_ids, small_product_ids, small_features])

# Transfer weights
old_user_weights = old_model.user_embedding.get_weights()[0]
old_product_weights = old_model.product_embedding.get_weights()[0]

new_user_weights = np.random.normal(size=(num_users, model.embedding_size))
for user, old_idx in old_user_id_mapping.items():
    if user in user_id_mapping:
        new_idx = user_id_mapping[user]
        new_user_weights[new_idx] = old_user_weights[old_idx]

new_product_weights = np.random.normal(size=(num_products, model.embedding_size))
for product, old_idx in old_product_id_mapping.items():
    if product in product_id_mapping:
        new_idx = product_id_mapping[product]
        new_product_weights[new_idx] = old_product_weights[old_idx]

# Set the embedding weights
model.user_embedding.set_weights([new_user_weights])
model.product_embedding.set_weights([new_product_weights])

# Transfer weights for other layers
for layer in model.layers:
    if layer.name not in ['user_embedding', 'product_embedding']:
        old_layer = old_model.get_layer(layer.name)
        layer.set_weights(old_layer.get_weights())

# Step 11: Compile and retrain the model
model.compile(optimizer='adam', loss='binary_crossentropy')
train_data = tf.data.Dataset.from_tensor_slices(((train_user_ids, train_product_ids, train_features), train_labels)).shuffle(10000).batch(512)
val_data = tf.data.Dataset.from_tensor_slices(((test_user_ids, test_product_ids, test_features), test_labels)).batch(512)
model.fit(train_data, epochs=10, validation_data=val_data)  # Fewer epochs for retraining

# Step 12: Save updated artifacts
model.save('app/hybrid_recommender_model.keras')
with open('app/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
with open('app/mlb.pkl', 'wb') as file:
    pickle.dump(mlb, file)
with open('app/user_id_mapping.pkl', 'wb') as file:
    pickle.dump(user_id_mapping, file)
with open('app/product_id_mapping.pkl', 'wb') as file:
    pickle.dump(product_id_mapping, file)
with open('app/product_style_dict.pkl', 'wb') as file:
    pickle.dump(product_style_dict, file)
with open('app/input_shape.pkl', 'wb') as file:
    pickle.dump(input_shape, file)