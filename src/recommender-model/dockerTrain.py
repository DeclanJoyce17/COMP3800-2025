import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import ast
import pickle

# Define numeric features
numeric_features = [
    'product_view_count', 'product_description_read_count','searched_product_count', 'product_favourite_count','product_purchase_count','product_link_open_count',
    'style_description_read_count', 'style_image_view_styleguide_count', 'style_image_view_content_count', 'product_added_to_cart_count',
    'style_list_open_count', 'purchased_item_review_count'
]

# Load data
data = pd.read_csv('app/train.csv')

# Convert product_styles strings to lists
def parse_styles(styles_str):
    try:
        return ast.literal_eval(styles_str)
    except Exception as e:
        print(f"Error parsing styles: {e}")
        return []

data['product_styles'] = data['product_styles'].apply(parse_styles)

# Normalize the numerical features
scaler = MinMaxScaler().fit(data[numeric_features])
data[numeric_features] = scaler.transform(data[numeric_features])

# Encode product styles using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
style_features = mlb.fit_transform(data['product_styles'])
num_styles = len(mlb.classes_)

# Save the number of styles
with open('app/num_styles.pkl', 'wb') as file:
    pickle.dump(num_styles, file)

# Prepare collaborative filtering data
user_ids = data['user_id'].values
product_ids = data['product_id'].values

# Get unique user and product IDs
unique_user_ids = np.unique(user_ids)
unique_product_ids = np.unique(product_ids)

# Get user and product mappings
user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
product_id_mapping = {product_id: idx for idx, product_id in enumerate(unique_product_ids)}

num_users = len(unique_user_ids)
num_products = len(unique_product_ids)

# Map user_ids and product_ids to indices
mapped_user_ids = np.array([user_id_mapping[uid] for uid in user_ids])
mapped_product_ids = np.array([product_id_mapping[aid] for aid in product_ids])

# Concatenate numerical and style features
numeric_features_matrix = data[numeric_features].values
full_features_matrix = np.hstack([numeric_features_matrix, style_features])

# Create positive labels (1s)
labels = np.ones(len(mapped_user_ids))

# Create negative samples
negative_samples = []

for user_id in unique_user_ids:
    positive_products = product_ids[user_ids == user_id]
    negative_products = np.setdiff1d(unique_product_ids, positive_products)
    neg_samples_count = min(10, len(positive_products), len(negative_products))
    if neg_samples_count > 0:
        sampled_neg_products = np.random.choice(negative_products, size=neg_samples_count, replace=False)
        for neg_product in sampled_neg_products:
            negative_samples.append([user_id_mapping[user_id], product_id_mapping[neg_product], 0])

# Combine positive and negative samples
positive_samples = np.column_stack((mapped_user_ids, mapped_product_ids, labels))
negative_samples = np.array(negative_samples)

if len(negative_samples) < len(positive_samples):
    additional_negative_samples = []
    while len(additional_negative_samples) < (len(positive_samples) - len(negative_samples)):
        user_id = np.random.choice(unique_user_ids)
        negative_products = np.setdiff1d(unique_product_ids, product_ids[user_ids == user_id])
        if len(negative_products) > 0:
            neg_product = np.random.choice(negative_products)
            additional_negative_samples.append([user_id_mapping[user_id], product_id_mapping[neg_product], 0])
    negative_samples = np.vstack((negative_samples, additional_negative_samples))

combined_samples = np.vstack((positive_samples, negative_samples))
np.random.shuffle(combined_samples)

train_samples, test_samples = train_test_split(combined_samples, test_size=0.2, random_state=42)

train_user_ids = train_samples[:, 0].astype(np.int32)
train_product_ids = train_samples[:, 1].astype(np.int32)
train_labels = train_samples[:, 2].astype(np.float32)

test_user_ids = test_samples[:, 0].astype(np.int32)
test_product_ids = test_samples[:, 1].astype(np.int32)
test_labels = test_samples[:, 2].astype(np.float32)

train_full_features = full_features_matrix[train_product_ids]
test_full_features = full_features_matrix[test_product_ids]

class HybridModel(tf.keras.Model):
    def __init__(self, num_users, num_products, num_numeric_features, num_styles, embedding_size=64, **kwargs):
        super(HybridModel, self).__init__(**kwargs)  # Pass kwargs to parent class to handle Keras' internal params
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

model = HybridModel(num_users, num_products, len(numeric_features), num_styles)
model.compile(optimizer='adam', loss='binary_crossentropy')

train_data = tf.data.Dataset.from_tensor_slices(((train_user_ids, train_product_ids, train_full_features), train_labels))
train_data = train_data.shuffle(10000).batch(512)

val_data = tf.data.Dataset.from_tensor_slices(((test_user_ids, test_product_ids, test_full_features), test_labels))
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
