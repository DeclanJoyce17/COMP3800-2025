import pandas as pd
import ast

data = pd.read_csv('/app/products.csv')

# Load interactions from CSV
def load_interactions():
    return pd.read_csv('/app/test_raw_interaction_data.csv', parse_dates=["timestamp"])

def popularity_score(product_id):
    interactions = load_interactions()
    
    # Filter interactions for purchases of the given product
    product_purchases = interactions[
        (interactions["product_id"] == product_id) & 
        (interactions["type"] == "purchase")
    ]
    
    purchase_count = product_purchases.shape[0]
    
    # Normalize the score, maxing out at 1.0
    return min(1.0, purchase_count / 100)

def price_score(user_id, product_id):
    interactions = load_interactions()
    
    # Get user-specific interactions
    user_interactions = interactions[interactions["user_id"] == user_id]

    # If no past interactions, return default score
    if user_interactions.empty:
        return 0.1  
    
    # Merge user interactions with product prices from `data` using product_id
    user_prices = user_interactions.merge(data[["id", "price"]], left_on="product_id", right_on="id", how="left")

    # Remove interactions without price information
    user_prices = user_prices.dropna(subset=["price"])

    # If still empty, return default score
    if user_prices.empty:
        return 0.1  

    # Get the average price from previous user interactions
    avg_price = user_prices["price"].mean()
    
    # Get the current product price
    current_price = data.loc[data["id"] == product_id, "price"].values

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

    # Merge user interactions with product categories from `data` using product_id
    user_categories = user_interactions.merge(data[["id", "productTypes"]], left_on="product_id", right_on="id", how="left")

    # Remove interactions without category information
    user_categories = user_categories.dropna(subset=["productTypes"])

    # If still empty, return default score
    if user_categories.empty:
        return 0.1  

    # Convert the string representation of a list to an actual list
    user_categories["productTypes"] = user_categories["productTypes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Expand the lists in the 'productTypes' column into individual rows
    expanded_categories = user_categories.explode("productTypes")

    # Get the frequency of categories the user has interacted with
    category_counts = expanded_categories["productTypes"].value_counts()

    # Get the category of the current product
    current_category = data.loc[data["id"] == product_id, "productTypes"].values

    # If product category is missing, return default score
    if len(current_category) == 0:
        return 0.1

    # Convert the product category string to a list (if it's in string format)
    current_category = ast.literal_eval(current_category[0]) if isinstance(current_category[0], str) else current_category[0]

    # Check if the current product category is in the list of categories the user has interacted with
    category_interactions = category_counts.get(current_category[0], 0)  # Taking the first category from the list for matching
    category_fraction = category_interactions / len(expanded_categories)

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
    # Get the availability status for the product from the 'availability' column in data
    availability_status = data.loc[data["id"] == product_id, "availability"].values

    # If product_id is not found or availability is not "IN_STOCK", return 0
    if len(availability_status) == 0 or availability_status[0] != "IN_STOCK":
        return 0
    
    return 1

def rerank_recommendations(user_id, recommendations):

    reranked_products = []
    length = (len(recommendations))

    for index, (product_id, placeholder) in enumerate(recommendations):
        price = price_score(user_id, product_id)
        category = category_score(user_id, product_id)
        popularity = popularity_score(product_id)
        #proVendor = vendor_score(product_id)
        #discount = discount_score(product_id)
        stock = stock_score(product_id)

        base_score = 0.0 * ((length - index)/length) 
        price_weighted = 1 * price
        category_weighted = 0.3 * category
        popularity_weighted = 0.2 * popularity
        #proVendor = 1 * proVendor
        #discount = 0.2 * discount
        stock = 0.5 * stock
        
        final_score = base_score + price_weighted + category_weighted + popularity_weighted + stock
        # + proVendor + discount 

        reranked_products.append((product_id, placeholder, final_score))
        
    reranked_products.sort(key=lambda x: x[2], reverse=True)

    # print("Reranked")
    # for product in reranked_products:
    #     print(product)
    
    return [(product[0], product[1]) for product in reranked_products]

