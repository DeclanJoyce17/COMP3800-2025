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
from rerank import rerank_recommendations  # Import reranking function
from filtering import filter_products, read_csv_to_dict, read_filters_from_json, map_recommended_products, read_csv


app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load necessary files
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

data = pd.read_csv('app/train.csv')

# CHECK: product_styles column should be a list of strings
print("product_styles type:", type(data['product_styles']))

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

def get_top_n_recommendations(user_id, num_products, n=num_products):
    user_idx = user_id_mapping.get(user_id, -1)
    if user_idx == -1:
        return []

    all_product_ids = np.arange(num_products, dtype=np.int32)
    user_tensor = tf.convert_to_tensor(np.full(shape=(num_products,), fill_value=user_idx, dtype=np.int32))
    product_tensor = tf.convert_to_tensor(all_product_ids, dtype=tf.int32)
    full_features_tensor = tf.convert_to_tensor(full_features_matrix, dtype=tf.float32)

    predictions = model.predict([user_tensor, product_tensor, full_features_tensor], batch_size=128)
    top_indices = np.argsort(predictions.flatten())[-n:][::-1]
    scores = predictions.flatten()[top_indices]

    recommendations = [list(product_id_mapping.keys())[i] for i in top_indices]
    return list(zip(recommendations, scores))

def turn_to_json(
    user_id, art_type, colors, color_or_and, from_my_favorites,
    materials, material_or_and, price_max, price_min, price_min_raw,
    product_types, seller_city, seller_country, seller_state,
    styles, style_or_and
):
    filters = {
        "user_id": user_id,
        "artType": art_type,
        "colors": colors,
        "colorOrAnd": color_or_and,
        "fromMyFavorites": from_my_favorites,
        "materials": materials,
        "materialOrAnd": material_or_and,
        "priceMax": price_max,
        "priceMin": float(price_min_raw) if price_min_raw else None,
        "productTypes": product_types,
        "sellerLocation": {
            "city": seller_city,
            "state": seller_state,
            "country": seller_country
        },
        "styles": styles,
        "styleOrAnd": style_or_and
    }

    # Save to filter.json
    with open("filter.json", "w") as f:
        json.dump(filters, f, indent=2)

    return filters


def decode_cursor(cursor):
    """Decode the cursor to get the offset."""
    if not cursor:
        return 0
    try:
        # Expected format: "{offset}_{hash}"
        offset = int(cursor.split('_')[0])
        return offset
    except (ValueError, IndexError):
        return 0


def encode_cursor(offset, last_score):
    """Create a cursor for the next page."""
    if offset >= len(product_id_mapping):
        return None
    # Create a cursor that includes both the offset and a hash of the last score
    cursor_string = f"{offset}_{hashlib.sha256(str(last_score).encode()).hexdigest()[:8]}"
    return cursor_string

@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id = request.args.get("user_id")
        
        #add here
        art_type = request.args.get("artType", "")
        colors = request.args.getlist("colors")  # ?colors=red&colors=blue
        color_or_and = request.args.get("colorOrAnd", "OR")
        from_my_favorites = request.args.get("fromMyFavorites", "false").lower() == "true"
        materials = request.args.getlist("materials")
        material_or_and = request.args.get("materialOrAnd", "OR")

        # Price range
        price_min_raw = request.args.get("priceMin")
        price_min = float(price_min_raw) if price_min_raw else None
        price_max = float(request.args.get("priceMax", 100000000000.0))

        # Product types
        product_types = request.args.getlist("productTypes")

        # Seller location (nested fields)
        seller_city = request.args.get("sellerLocation.city", "")
        seller_state = request.args.get("sellerLocation.state", "")
        seller_country = request.args.get("sellerLocation.country", "")

        # Styles
        styles = request.args.getlist("styles")
        style_or_and = request.args.get("styleOrAnd", "OR")
        
        turn_to_json(user_id, art_type, colors, color_or_and, from_my_favorites, materials,
                                material_or_and, price_max, price_min, price_min_raw, product_types,
                                  seller_city, seller_country, seller_state, styles, style_or_and)

        filters = read_filters_from_json("filter.json")
        #print(filters)
        #print(user_id)
        #print(filters)
        seller_locations = read_csv_to_dict("app/sellerLocation.csv", "sellerId")
        #print(seller_locations)
        limit = int(request.args.get("limit", 24))
        cursor = request.args.get("cursor")
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        # Initialize recommendations in session if not present
        if "recommendations" not in session or not cursor:
            session["recommendations"] = get_top_n_recommendations(user_id, num_products, n=100)
            session["seen"] = set()
            

        recommendations = session["recommendations"]

        products = read_csv("app/products.csv")
        print(recommendations)
        recommendations2 = filter_products(products, filters, seller_locations)
        #print(recommendations2)
        recommendations3 = map_recommended_products(recommendations2, recommendations)
        print("Recommendations 3: ")
        print(recommendations3)
        seen = session["seen"]
        #recommendations = recommendations3
        # for product in recommendations:
        #     print(product)

        reranked_recommendation = rerank_recommendations(user_id, recommendations3)

        # print("Reranked Recommendations:")

        # for product in reranked_recommendation:
        #     print(product)

        # Use pagination module
        paginated_product_ids, next_cursor = paginate_recommendations(reranked_recommendation, seen, cursor, limit)
        # print(paginated_product_ids)

        # Update seen products
        seen.update(paginated_product_ids)
        session["seen"] = seen
        session.modified = True
        
        return jsonify({
            "user_id": user_id,
            "limit": limit,
            "recommendations": paginated_product_ids,
            "cursor": next_cursor,
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

@app.route('/filtered_data', methods=['GET'])
def get_data():
    try:
        # Load the CSV file
        df = pd.read_csv('filtered_products.csv')  # Change 'data.csv' to your actual file
        
        # Convert DataFrame to JSON
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)