from flask import Flask, request, jsonify, session
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import ast
import json
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from flask_session import Session
from pagination import paginate_recommendations  # Import the pagination logic

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load necessary files
with open('app/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('app/mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

with open('app/num_styles.pkl', 'rb') as file:
    num_styles = pickle.load(file)

data = pd.read_csv('app/train.csv')

# Convert styles from strings to lists
def parse_styles(styles_str):
    try:
        return ast.literal_eval(styles_str)
    except Exception:
        return []

data['product_styles'] = data['product_styles'].apply(parse_styles)

numeric_features = [
    'product_view_count', 'product_description_read_count', 'searched_product_count',
    'product_favourite_count', 'product_purchase_count', 'product_link_open_count',
    'style_description_read_count', 'style_image_view_styleguide_count',
    'style_image_view_content_count', 'product_added_to_cart_count',
    'style_list_open_count', 'purchased_item_review_count'
]
data[numeric_features] = scaler.transform(data[numeric_features])
style_features = mlb.transform(data['product_styles'])
full_features_matrix = np.hstack([data[numeric_features].values, style_features])

user_id_mapping = {user_id: idx for idx, user_id in enumerate(data['user_id'].unique())}
product_id_mapping = {product_id: idx for idx, product_id in enumerate(data['product_id'].unique())}
num_users = len(user_id_mapping)
num_products = len(product_id_mapping)

full_features_matrix = np.tile(full_features_matrix, (num_products, 1))


class HybridModel(tf.keras.Model):
    def __init__(self, num_users, num_products, num_numeric_features, num_styles, embedding_size=64, **kwargs):
        super(HybridModel, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_products = num_products
        self.num_numeric_features = num_numeric_features
        self.num_styles = num_styles
        self.embedding_size = embedding_size

        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal')
        self.product_embedding = tf.keras.layers.Embedding(num_products, embedding_size,
                                                           embeddings_initializer='he_normal')
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


model = tf.keras.models.load_model('app/hybrid_recommender_model.keras', custom_objects={"HybridModel": HybridModel})


def get_top_n_recommendations(user_id, num_products, n=100):
    user_idx = user_id_mapping.get(user_id, -1)
    if user_idx == -1:
        return []

    all_product_ids = np.arange(num_products, dtype=np.int32)
    user_tensor = tf.convert_to_tensor(np.full(shape=(num_products,), fill_value=user_idx, dtype=np.int32))
    product_tensor = tf.convert_to_tensor(all_product_ids, dtype=tf.int32)
    full_features_tensor = tf.convert_to_tensor(full_features_matrix[:num_products], dtype=tf.float32)

    predictions = model.predict([user_tensor, product_tensor, full_features_tensor], batch_size=128)
    top_indices = np.argsort(predictions.flatten())[-n:][::-1]
    scores = predictions.flatten()[top_indices]

    recommendations = [list(product_id_mapping.keys())[i] for i in top_indices]
    return list(zip(recommendations, scores))


@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id = request.args.get("user_id")
        limit = int(request.args.get("limit", 24))
        cursor = request.args.get("cursor")

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        # Initialize recommendations in session if not present
        if "recommendations" not in session or not cursor:
            session["recommendations"] = get_top_n_recommendations(int(user_id), num_products, n=100)
            session["seen"] = set()

        recommendations = session["recommendations"]
        seen = session["seen"]

        # Use pagination module
        paginated_product_ids, next_cursor = paginate_recommendations(recommendations, seen, cursor, limit)

        # Update seen products
        seen.update(paginated_product_ids)
        session["seen"] = seen
        session.modified = True

        return jsonify({
            "user_id": user_id,
            "limit": limit,
            "recommendations": paginated_product_ids,
            "cursor": next_cursor
        })

    except ValueError:
        return jsonify({"error": "Invalid user_id or cursor"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500




# TESTING FILTERING STUFF BELOW
@app.route("/filters", methods=["POST"])
def save_filters():
    try:
        # Get JSON data from the request
        filters = request.get_json()
        
        if not filters:
            return jsonify({"error": "No filters provided"}), 400

        # Save filters to a JSON file
        with open("filters.json", "w") as json_file:
            json.dump(filters, json_file, indent=4)
        
        # Return the exact JSON structure received
        return jsonify(filters)
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)