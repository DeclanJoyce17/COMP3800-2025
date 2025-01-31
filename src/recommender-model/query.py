import os
import requests
import csv
import socket
from dotenv import load_dotenv

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

if ip_address.startswith("127.") or ip_address.startswith("192.168.") or ip_address.startswith("10."):
    prisma_server_url = os.getenv('LOCAL_PRISMA_SERVER_URL')
else:
    prisma_server_url = os.getenv('EC2_PRISMA_SERVER_URL')

print(f"Using Prisma REST API URL: {prisma_server_url}")

# Function to fetch all interaction events using Prisma REST API
def fetch_all_interaction_events():
    response = requests.get(f"{prisma_server_url}/interactionEvents")
    response.raise_for_status()
    return response.json()

# Aggregation logic
def aggregate_interaction_events(interaction_events):
    aggregated_data = {}

    for event in interaction_events:
        session_user = event.get("sessionUser")
        user_id = session_user["id"] if session_user else None
        product = event.get("product")
        style = event.get("style")
        interaction_type = event["type"]

        if style:
            product_ids = [style_node["product"]["id"] for style_node in style.get("productStyles", []) if style_node.get("product")]
            for product_id in product_ids:
                product_details = [node for node in style.get("productStyles", []) if node.get("product") and node["product"]["id"] == product_id]
                product_styles = [style_node["style"]["name"] for product_detail in product_details for style_node in product_detail["product"].get("productStyles", []) if style_node.get("style")]
                aggregate_data(aggregated_data, user_id, product_id, interaction_type, product_styles)

        if product:
            product_id = product["id"]
            product_styles = [style_node["style"]["name"] for style_node in product.get("productStyles", []) if style_node.get("style")]
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
            "style_description_read_count": 0,
            "style_image_view_styleguide_count": 0,
            "style_image_view_content_count": 0,
            "style_list_open_count": 0,
            "purchased_item_review_count": 0,
            "product_styles": set()
        }

    interaction_map = {
        "product_DESCRIPTION_READ": "product_description_read_count",
        "product_LINK_OPEN": "product_link_open_count",
        "STYLE_DESCRIPTION_READ": "style_description_read_count",
        "STYLE_IMAGE_VIEW_STYLEGUIDE": "style_image_view_styleguide_count",
        "STYLE_IMAGE_VIEW_CONTENT": "style_image_view_content_count",
        "STYLE_LIST_OPEN": "style_list_open_count"
    }

    if interaction_type in interaction_map:
        data[key][interaction_map[interaction_type]] += 1

    data[key]["product_styles"].update(product_styles)

# Function to save data as CSV
def save_data_as_csv(data, filename):
    fieldnames = [
        "user_id", "product_id", "searched_product_count", "product_description_read_count",
        "product_link_open_count", "style_description_read_count", "style_image_view_styleguide_count",
        "style_image_view_content_count", "style_list_open_count", "purchased_item_review_count",
        "product_styles"
    ]

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()

        for key, value in data.items():
            value["product_styles"] = f"['{', '.join(value['product_styles'])}']"
            writer.writerow(value)

# Main function to run the process
def main():
    interaction_events = fetch_all_interaction_events()
    aggregated_data = aggregate_interaction_events(interaction_events)
    
    filename = 'app/retrain.csv' if os.path.exists('app/train.csv') else 'app/train.csv'
    save_data_as_csv(aggregated_data, filename)

if __name__ == "__main__":
    main()
