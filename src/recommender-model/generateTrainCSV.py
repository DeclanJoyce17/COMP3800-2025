import pandas as pd
import random
import uuid
from io import StringIO

# Load products.csv into a DataFrame
products_df = pd.read_csv('/app/products.csv')

# Specified seller ID
specified_seller_id = "fe5b5102-1003-4811-ada4-04f543a044d9"

# Filter products by seller
specified_products = products_df[products_df['sellerId'] == specified_seller_id]
other_products = products_df[products_df['sellerId'] != specified_seller_id]

# Generate unique user IDs
num_users = 20
users = [str(uuid.uuid4()) for _ in range(num_users)]

# Function to generate interaction data
def generate_interactions(is_high):
    if is_high:
        return {
            'searched_product_count': random.randint(2, 5),
            'product_description_read_count': random.randint(2, 5),
            'product_link_open_count': random.randint(2, 5),
            'style_description_read_count': random.randint(1, 3),
            'style_image_view_styleguide_count': random.randint(1, 3),
            'style_image_view_content_count': random.randint(1, 3),
            'style_list_open_count': random.randint(1, 3),
            'purchased_item_review_count': 1 if random.random() < 0.5 else 0,
            'product_favourite_count': random.randint(1, 2),
            'product_added_to_cart_count': random.randint(1, 2),
            'product_view_count': random.randint(5, 10),
            'product_purchase_count': 1 if random.random() < 0.7 else 0
        }
    else:
        return {
            'searched_product_count': random.randint(0, 2),
            'product_description_read_count': random.randint(0, 2),
            'product_link_open_count': random.randint(0, 2),
            'style_description_read_count': random.randint(0, 1),
            'style_image_view_styleguide_count': random.randint(0, 1),
            'style_image_view_content_count': random.randint(0, 1),
            'style_list_open_count': random.randint(0, 1),
            'purchased_item_review_count': 1 if random.random() < 0.1 else 0,
            'product_favourite_count': random.randint(0, 1),
            'product_added_to_cart_count': random.randint(0, 1),
            'product_view_count': random.randint(1, 3),
            'product_purchase_count': 1 if random.random() < 0.3 else 0
        }

# Generate interactions
interactions = []
for user in users:
    # Sample products: 7 from specified seller, 3 from others
    specified_sample = specified_products.sample(7, replace=True) if len(specified_products) < 7 else specified_products.sample(7)
    other_sample = other_products.sample(3, replace=True) if len(other_products) < 3 else other_products.sample(3)
    selected_products = pd.concat([specified_sample, other_sample])
    
    for _, row in selected_products.iterrows():
        product_id = row['id']
        is_high = row['sellerId'] == specified_seller_id
        interaction = generate_interactions(is_high)
        interaction['user_id'] = user
        interaction['product_id'] = product_id
        interaction['product_styles'] = row['styles']
        interactions.append(interaction)

# Create DataFrame
train_df = pd.DataFrame(interactions)

# Define column order to match your train.csv
column_order = [
    'user_id', 'product_id', 'searched_product_count', 'product_description_read_count',
    'product_link_open_count', 'style_description_read_count', 'style_image_view_styleguide_count',
    'style_image_view_content_count', 'style_list_open_count', 'purchased_item_review_count',
    'product_favourite_count', 'product_added_to_cart_count', 'product_styles',
    'product_view_count', 'product_purchase_count'
]

# Reorder columns
train_df = train_df[column_order]

# Save to CSV
train_df.to_csv('/app/train.csv', index=False)

print("New train.csv has been generated successfully with products from seller 'fe5b5102-1003-4811-ada4-04f543a044d9' having more positive interactions!")