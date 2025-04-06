import csv
import json

def matches_criteria(value_list, filter_list, condition="OR"):
    """Applies OR/AND filtering logic correctly."""
    if not filter_list:  # If no filter is provided, accept all
        return True
    
    value_set = set(map(str.lower, value_list))
    filter_set = set(map(str.lower, filter_list))
    
    if condition.upper() == "AND":
        return all(item in value_set for item in filter_set)  # Must contain all filter values
    elif condition.upper() == "OR":
        return any(item in value_set for item in filter_set)  # Must contain at least one filter value

    return False  # Default case (should not happen)


def read_csv_to_dict(filename, key_column):
    """Reads a CSV file and returns a dictionary mapped by key_column."""
    data = {}
    with open(filename, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[row[key_column]] = row  # Map by key column
    return data

def parse_list(value):
    """Converts a string representation of a list into an actual list."""
    try:
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):  # JSON format
            return json.loads(value.replace("'", '"'))
        return [v.strip() for v in value.split(",")]  # Handle comma-separated values
    except json.JSONDecodeError:
        return []

def clean_list(value_list):
    """Ensures all list values are cleaned and properly formatted."""
    cleaned_list = []
    for value in value_list:
        if isinstance(value, str):
            cleaned_list.extend([v.strip() for v in value.split(",") if v.strip()])
        else:
            cleaned_list.append(value)  # In case it's already a list
    return cleaned_list

def filter_products(products, filters, seller_locations):
    """Filters products based on criteria from filters and returns a list of product IDs."""
    filtered_product_ids = []

    for product in products:
        # Parse and clean necessary fields
        art_types = clean_list(parse_list(product.get("artTypes", "")))
        colors = clean_list(parse_list(product.get("colors", "")))
        materials = clean_list(parse_list(product.get("materials", "")))
        styles = clean_list(parse_list(product.get("styles", "")))

        try:
            price = int(product["price"])
        except (ValueError, KeyError):
            price = 0

        seller_id = product["sellerId"]
        seller_info = seller_locations.get(seller_id, {})

        if filters.get("sellerLocation"):
            location_filter = filters["sellerLocation"]
            if location_filter.get("city") and location_filter["city"].lower() != seller_info.get("city", "").lower():
                continue
            if location_filter.get("state") and location_filter["state"].lower() != seller_info.get("state", "").lower():
                continue
            if location_filter.get("country") and location_filter["country"].lower() != seller_info.get("country", "").lower():
                continue

        if filters.get("artType") and filters["artType"].lower() not in map(str.lower, art_types):
            continue

        if not matches_criteria(colors, filters.get("colors", []), filters.get("colorOrAnd", "OR")):
            continue

        if not matches_criteria(materials, filters.get("materials", []), filters.get("materialOrAnd", "OR")):
            continue

        if not matches_criteria(styles, filters.get("styles", []), filters.get("styleOrAnd", "OR")):
            continue

        if filters.get("priceMin") is not None and price < filters["priceMin"]:
            continue
        if filters.get("priceMax") is not None and price > filters["priceMax"]:
            continue
        filtered_product_ids.append(product["id"])
    #print(filtered_product_ids)
    return filtered_product_ids


def map_recommended_products(filtered_ids, recommended_list):
    """
    Maps filtered product IDs to recommended product scores.

    Parameters:
    - filtered_ids: List of product IDs returned from filtering
    - recommended_list: List of tuples (product_id, score)

    Returns:
    - A list of tuples (product_id, score) for matching recommended products
    """
    filtered_set = set(filtered_ids)  # For quick lookup
    mapped = [
        (prod_id, score)
        for prod_id, score in recommended_list
        if prod_id in filtered_set
    ]
    return mapped


def read_csv(filename):
    """Reads a CSV file and returns a list of dictionaries."""
    with open(filename, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

import json

def read_filters_from_json(filepath):
    """
    Read filters from a JSON file and ensure it returns a dictionary.
    """
    with open(filepath, 'r') as f:
        filters = json.load(f)
    
    if isinstance(filters, dict):
        return filters
    else:
        raise ValueError("Filters file does not contain a valid dictionary.")




def write_filtered_slugs_to_csv(filtered_products, user_id, filename="filtered_products.csv"):
    """Writes the filtered product slugs along with the user ID to a CSV file, overwriting each time."""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["User_ID", "Filtered_Products"])  # Header
        
        filtered_slugs = [product["id"] for product in filtered_products]
        formatted_slugs = str(filtered_slugs)  # Convert list to string directly
        
        writer.writerow([user_id, formatted_slugs])  # Write user ID and formatted list of slugs


# Load data
products = read_csv("app/products.csv")  # CSV containing product data
filters = read_filters_from_json("filter.json")  # JSON containing filters
seller_locations = read_csv_to_dict("app/sellerLocation.csv", "sellerId")
# Apply filtering
filtered_results = filter_products(products, filters, seller_locations)

#write_filtered_slugs_to_csv(filtered_results, user_id)

 # Output results
for product in filtered_results:
    print(product)
