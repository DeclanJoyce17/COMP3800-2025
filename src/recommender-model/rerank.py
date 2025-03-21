import pandas as pd
from app import data, interaction_file  # Import necessary components

# Load interactions from CSV
def load_interactions():
    return pd.read_csv(interaction_file, parse_dates=["timestamp"])

def popularity_score(product_id):
    interactions = load_interactions()
    
    # Filter interactions for purchases of the given product
    product_purchases = interactions[
        (interactions["product_id"] == product_id) & 
        (interactions["event_type"] == "purchase")
    ]
    
    purchase_count = product_purchases.shape[0]
    
    # Normalize the score, maxing out at 1.0
    return min(1.0, purchase_count / 1000)

def price_score(user_id, product_id):
    interactions = load_interactions()
    
    # Get user-specific interactions
    user_interactions = interactions[interactions["user_id"] == user_id]

    # If no past interactions, return default score
    if user_interactions.empty:
        return 0.1  

    # Get the average price from previous user interactions
    avg_price = user_interactions["product_price"].mean()
    
    # Get the current product price
    current_price = data.loc[data["product_id"] == product_id, "product_price"].values

    # If product price is missing, return default score
    if len(current_price) == 0:
        return 0.1

    # Calculate the absolute price difference and normalize the score
    price_diff = abs(current_price[0] - avg_price) / avg_price
    adjusted_score = 1.0 - price_diff
    
    # Return adjusted score, with a minimum of 0.1
    return max(0.1, adjusted_score)

def category_score(user_id, product_id):
    interactions = load_interactions()
    
    # Get user-specific interactions
    user_interactions = interactions[interactions["user_id"] == user_id]

    # If no past interactions, return default score
    if user_interactions.empty:
        return 0.1

    # Get the categories the user has interacted with
    user_categories = user_interactions["product_category"].value_counts()

    # Get the category of the current product
    current_category = data.loc[data["product_id"] == product_id, "product_category"].values

    # If product category is missing, return default score
    if len(current_category) == 0:
        return 0.1

    # Calculate the fraction of interactions with the same category
    category_interactions = user_categories.get(current_category[0], 0)
    category_fraction = category_interactions / len(user_interactions)

    # Return the score based on the fraction of interactions
    return max(0.1, category_fraction)


def vendor_score(product_id):
    interactions = load_interactions()
    
    # Check if the product is from a pro account vendor (pro_vendor column)
    pro_vendor = interactions[interactions["product_id"] == product_id]["pro_vendor"].values
    
    # Return 1 if the vendor is a pro account (pro_vendor == 1), otherwise return 0
    if len(pro_vendor) > 0 and pro_vendor[0] == 1:
        return 1
    
    return 0

def discount_score(product_id):
    interactions = load_interactions()
    
    # Get the discount status for the given product
    discount_status = interactions.loc[interactions["product_id"] == product_id, "discount"].values
    
    # If product has no discount information, return 0
    if len(discount_status) == 0 or discount_status[0] == 0:
        return 0
    
    return 1

def stock_score(product_id):
    interactions = load_interactions()
    
    # Get the stock status for the product from the 'instock' column
    instock_status = interactions.loc[interactions["product_id"] == product_id, "instock"].values
    
    # If no stock status or the product is out of stock, return a default score of 0
    if len(instock_status) == 0 or instock_status[0] == 0:
        return 0
    
    return 1

def rerank_recommendations(user_id, recommendations):

    reranked_products = []
    length = (len(recommendations))

    for index, product in enumerate(recommendations):
        price = price_score(user_id, product)
        category = category_score(user_id, product)
        popularity = popularity_score(product)
        proVendor = vendor_score(product)
        discount = discount_score(product)
        stock = stock_score(product)

        # Multiply the base score by (10 - index), where index is the position in the recommendations
        base_score = 1 * ((length - index)/length) 
        price_weighted = 0.3 * price
        category_weighted = 0.3 * category
        popularity_weighted = 0.2 * popularity
        proVendor = 1 * proVendor
        discount = 0.2 * discount
        stock = 0.5 * stock
        
        # Compute the final score with weights for each factor
        final_score = base_score + price_weighted + category_weighted + popularity_weighted + proVendor + discount + stock

        reranked_products.append((product, final_score))

    # Sort products by final score in descending order
    reranked_products.sort(key=lambda x: x[1], reverse=True)
    
    return [p[0] for p in reranked_products]

