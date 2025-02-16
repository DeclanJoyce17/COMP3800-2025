import json
import csv

#load product data from train.csv
def load_product_data(csv_filename):
    product_data = {}
    with open(csv_filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            product_id = int(row["product_id"])
            product_colors = row["product_colors"].strip("[]").replace("'", "").split(", ")  #convert to list
            product_data[product_id] = product_colors
    return product_data

#map recommendations to product details with optional color filtering
def map_recommendations(json_filename, product_data, filter_colors=None):
    with open(json_filename, mode='r', encoding='utf-8') as file:
        user_recommendations = json.load(file)

    filtered_recommendations = []

    for user in user_recommendations:
        filtered_products = [
            product_id for product_id in user["Top_Recommendations"]
            if filter_colors is None or any(color in filter_colors for color in product_data.get(product_id, []))
        ]

        #add user only if they have filtered recommendations
        if filtered_products:
            filtered_recommendations.append({
                "User_ID": user["User_ID"],
                "Top_Recommendations": filtered_products
            })

    return filtered_recommendations

#save recommendations
def save_filtered_recommendations(output_filename, filtered_recommendations):
    with open(output_filename, mode='w', encoding='utf-8') as file:
        json.dump(filtered_recommendations, file, indent=4)

#main function
def main():
    csv_filename = "app/train.csv"
    json_filename = "app/user_recommendations.json"
    output_filename = "app/filtered_recommendations.json"

    
    filter_colors = ["Yellow", "Red"]

    product_data = load_product_data(csv_filename)
    filtered_recommendations = map_recommendations(json_filename, product_data, filter_colors)
    save_filtered_recommendations(output_filename, filtered_recommendations)

    print(f"Filtered recommendations saved to {output_filename}")

if __name__ == "__main__":
    main()
