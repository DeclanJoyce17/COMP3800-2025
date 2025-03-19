import csv
import json

def matches_criteria(value_list, filter_list, condition):
    """ Helper function to apply OR/AND filtering logic """
    if not filter_list:  # If filter is empty, accept all
        return True
    value_set = set(map(str.lower, value_list))
    filter_set = set(map(str.lower, filter_list))
    
    if condition == "AND":
        return filter_set.issubset(value_set)  # All filter values must be present
    return bool(value_set & filter_set)  # At least one must match (OR)

def read_products_from_csv(filename):
    """ Reads products from a CSV file and returns them as a list of dictionaries """
    products = {}
    with open(filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            products[row["id"]] = row  # Map product ID to product data
    return products

def read_filters_from_json(filename):
    """ Reads filter criteria from a JSON file """
    with open(filename, mode="r") as file:
        filters = json.load(file)
    return filters

def read_recommended_slugs(filename):
    """ Reads the top 5 recommended slugs for each user from the CSV file """
    user_recommendations = {}
    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_id = row["User_ID"]
            recommended_slugs = json.loads(row["Top_Recommendations"].replace("'", '"'))
            user_recommendations[user_id] = recommended_slugs
    return user_recommendations

def filter_products(products, filters, user_id, user_recommendations):
    """ Filters products based on the given criteria, skipping filters if they are blank or null """
    filtered_products = []

    recommended_slugs = user_recommendations.get(user_id, [])
    
    for slug in recommended_slugs:
        product = products.get(slug)  # Get product data using slug as ID
        if not product:
            continue  # Skip if product data is not found for the slug
        
        print(product['id'] + '\n')
        # Parse necessary fields from product data
        art_types = json.loads(product["artTypes"].replace("'", '"'))
        colors = json.loads(product["colors"].replace("'", '"'))
        materials = json.loads(product.get("materials", "[]").replace("'", '"'))
        product_types = json.loads(product["productTypes"].replace("'", '"'))
        styles = json.loads(product["styles"].replace("'", '"'))
        price = int(product["price"])  # Convert price from string to int

        # Apply filters only if they are not blank or null
        if filters.get("artType") and filters["artType"].lower() not in map(str.lower, art_types):
            continue  # Skip if art type does not match, unless filter is blank

        if filters.get("colors") and not matches_criteria(colors, filters["colors"], filters.get("colorOrAnd", "OR")):
            continue  # Skip if color filter does not match

        if filters.get("materials") and not matches_criteria(materials, filters["materials"], filters.get("materialOrAnd", "OR")):
            continue  # Skip if material filter does not match

        if filters.get("productTypes") and not matches_criteria(product_types, filters["productTypes"], "OR"):
            continue  # Always OR for productTypes
        
        if filters.get("styles") and not matches_criteria(styles, filters["styles"], filters.get("styleOrAnd", "AND")):
            continue  # Skip if style filter does not match

        if filters.get("priceMin") and filters.get("priceMax") and not (filters["priceMin"] <= price <= filters["priceMax"]):
            continue  # Skip if price is out of range
        
        # If all filters pass and the product is in the recommendations, add it to results
        filtered_products.append(product)

    return filtered_products

def write_filtered_slugs_to_csv(filtered_products, user_id, filename="filtered_products.csv"):
    """ Writes the user ID and their filtered products slugs to a CSV file """
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write user_id and their filtered products
        filtered_slugs = [product["id"] for product in filtered_products]
        
        # If there are filtered products, write them to the file
        if filtered_slugs:
            writer.writerow([user_id, str(filtered_slugs)])

# Example usage

# Read products, filters, and user recommendations from their respective files
products = read_products_from_csv("app/products.csv")  # Read product data and map by ID
filters = read_filters_from_json("filter.json")
user_recommendations = read_recommended_slugs("app/top_5_recommendations.csv")

# Specify a user ID (e.g., 1) to filter products based on their top recommendations
user_id = "1"

# Filter products based on the provided filters and top 5 recommended slugs for the user
filtered_results = filter_products(products, filters, user_id, user_recommendations)

# Write the filtered products slugs along with the user ID to the CSV file
write_filtered_slugs_to_csv(filtered_results, user_id)

# Optionally, print the filtered slugs to the console for verification
for product in filtered_results:
    print(product["id"])
