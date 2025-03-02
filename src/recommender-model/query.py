import os
import requests
import csv
import socket
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Determine environment based on the IP address
ip_address = get_ip()
print(f"Detected IP address: {ip_address}")

if ip_address.startswith("127.") or ip_address.startswith("192.168.") or ip_address.startswith("10.") or ip_address.startswith("172."):
    rest_api_url = os.getenv('LOCAL_REST_API_URL')
else:
    rest_api_url = os.getenv('EC2_REST_API_URL')

print(f"Using REST API URL: {rest_api_url}")

# Function to fetch interaction events from REST API
def fetch_all_interaction_events():
    response = requests.get(f"{rest_api_url}/interaction-event")
    response.raise_for_status()
    result = response.json()
    return result['interactionEvents']

# Aggregation logic
def aggregate_interaction_events(interaction_events):
    aggregated_data = {}

    for event in interaction_events:
        user_id = event.get("sessionUser", {}).get("id")
        product_id = event.get("product", {}).get("id")
        interaction_type = event["type"]
        product_styles = [style["name"] for style in event.get("product", {}).get("productStyles", [])]

        aggregate_data(aggregated_data, user_id, product_id, interaction_type, product_styles)
    
    return aggregated_data

# Function to aggregate data
def aggregate_data(data, user_id, product_id, interaction_type, product_styles):
    if not user_id or not product_id:
        return

    key = f"{user_id}-{product_id}"
    if key not in data:
        data[key] = {
            "user_id": user_id,
            "product_id": product_id,
            "searched_product_count": 0,
            "product_description_read_count": 0,
            "product_link_open_count": 0,
            "product_favourite_count": 0,
            "product_purchase_count": 0,
            "product_added_to_cart_count": 0,
            "style_description_read_count": 0,
            "style_image_view_styleguide_count": 0,
            "style_image_view_content_count": 0,
            "style_list_open_count": 0,
            "purchased_item_review_count": 0,
            "product_styles": set()
        }

    if interaction_type == "product_DESCRIPTION_READ":
        data[key]["product_description_read_count"] += 1
    elif interaction_type == "product_SEARCHED":
        data[key]['searched_product_count'] += 2
    elif interaction_type == "product_FAVOURITE":
        data[key]["product_favourite_count"] += 3
    elif interaction_type == "product_PURCHASE":
        data[key]["product_purchase_count"] += 5
    elif interaction_type == "product_ADDED_TO_CART":
        data[key]["product_added_to_cart_count"] += 3
    elif interaction_type == "product_LINK_OPEN":
        data[key]["product_link_open_count"] += 2
    elif interaction_type == "STYLE_DESCRIPTION_READ":
        data[key]["style_description_read_count"] += 1
    elif interaction_type == "STYLE_IMAGE_VIEW_STYLEGUIDE":
        data[key]["style_image_view_styleguide_count"] += 1
    elif interaction_type == "STYLE_IMAGE_VIEW_CONTENT":
        data[key]["style_image_view_content_count"] += 1
    elif interaction_type == "STYLE_LIST_OPEN":
        data[key]["style_list_open_count"] += 1
    
    data[key]["product_styles"].update(product_styles)

# Function to save interaction data as CSV
def save_data_as_csv(data, filename):
    fieldnames = [
        "user_id", "product_id", "searched_product_count", "product_description_read_count",
        "product_link_open_count", "style_description_read_count", "style_image_view_styleguide_count",
        "style_image_view_content_count", "style_list_open_count", "purchased_item_review_count",
        "product_styles", "product_favourite_count","product_purchase_count",
        "product_added_to_cart_count",
    ]

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()

        for key, value in data.items():
            value["product_styles"] = f"['{', '.join(value['product_styles'])}']"
            writer.writerow(value)

# Function to fetch products from REST API (could be changed to only fetch all new products since last update)
def fetch_all_products():
    response = requests.get(f"{rest_api_url}/products/recommender-data")
    response.raise_for_status()
    result = response.json()
    return result['products']

# Main function to run the process
def main():
    interaction_events = fetch_all_interaction_events()
    aggregated_data = aggregate_interaction_events(interaction_events)
    products = fetch_all_products()
    # print(json.dumps(products, indent=2))
    # print(len(products))
    
    filename = 'app/retrain.csv' if os.path.exists('app/train.csv') else 'app/train.csv'
    save_data_as_csv(aggregated_data, filename)

if __name__ == "__main__":
    main()