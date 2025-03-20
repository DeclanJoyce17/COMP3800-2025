import sys
import os
import pytest
import csv
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from unittest.mock import patch, MagicMock
from query import fetch_all_interaction_events
from query import (
    fetch_all_interaction_events, 
    fetch_all_products, 
    aggregate_interaction_events, 
    aggregate_data, 
    save_interaction_data_as_csv, 
    save_product_data_as_csv
)

@pytest.fixture
def mock_interaction_events():
  return [
  {
    "user_id": "a1b",
    "product_id": "bd1fc40e-5da6-4548-9931-ccd2dcf8a004",
    "searched_product_count": 2,
    "product_description_read_count": 1,
    "product_link_open_count": 3,
    "style_description_read_count": 1,
    "style_image_view_styleguide_count": 0,
    "style_image_view_content_count": 0,
    "style_list_open_count": 1,
    "purchased_item_review_count": 0,
    "product_favourite_count": 1,
    "product_added_to_cart_count": 1,
    "product_styles": "['Bohemian']",
    "product_view_count": 6,
    "product_purchase_count": 1
  },
  {
    "user_id": "a2b",
    "product_id": "bd1fc40e-5da6-4548-9931-ccd2dcf8a004",
    "searched_product_count": 0,
    "product_description_read_count": 0,
    "product_link_open_count": 1,
    "style_description_read_count": 2,
    "style_image_view_styleguide_count": 1,
    "style_image_view_content_count": 0,
    "style_list_open_count": 0,
    "purchased_item_review_count": 1,
    "product_favourite_count": 1,
    "product_added_to_cart_count": 0,
    "product_styles": "['Bohemian']",
    "product_view_count": 4,
    "product_purchase_count": 0
  }
]

@patch("query.requests.get") 
def test_fetch_all_interaction_events(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
      "interactionEvents": [
        {
          "user_id": "a1b",
          "product_id": "bd1fc40e-5da6-4548-9931-ccd2dcf8a004",
          "searched_product_count": 2,
          "product_description_read_count": 1,
          "product_link_open_count": 3,
          "style_description_read_count": 1,
          "style_image_view_styleguide_count": 0,
          "style_image_view_content_count": 0,
          "style_list_open_count": 1,
          "purchased_item_review_count": 0,
          "product_favourite_count": 1,
          "product_added_to_cart_count": 1,
          "product_styles": "['Bohemian']",
          "product_view_count": 6,
          "product_purchase_count": 1
        }
      ]
    }
   
    mock_response.status_code = 200
    mock_get.return_value = mock_response 

    result = fetch_all_interaction_events()

    assert isinstance(result, list)  
    assert len(result) == 1  
    assert result[0]["user_id"] == "a1b"  

    print("Test result:", result)

    def test_fetch_all_products():
      result = fetch_all_products()
      print("fetch_all_products Output:", result)
      
      assert isinstance(result, list), "Expected a list of products"
      assert all("product_id" in product for product in result), "Each product should have a product_id"

@patch("query.fetch_all_interaction_events")
def test_aggregate_interaction_events(mock_fetch):
  mock_fetch.return_value = [
    {
        "user_id": "a1b",
        "product_id": "bd1fc40e-5da6-4548-9931-ccd2dcf8a004",
        "type": "view",  
        "searched_product_count": 2,
        "product_description_read_count": 1,
        "product_link_open_count": 3,
        "product_purchase_count": 1,
    },
    {
        "user_id": "a1b",
        "product_id": "bd1fc40e-5da6-4548-9931-ccd2dcf8a004",
        "type": "click", 
        "searched_product_count": 1,
        "product_description_read_count": 0,
        "product_link_open_count": 1,
        "product_purchase_count": 0,
    },
]
  
  result = aggregate_interaction_events(mock_fetch.return_value)

  assert result == {
        "bd1fc40e-5da6-4548-9931-ccd2dcf8a004": {
            "searched_product_count": 3,
            "product_description_read_count": 1,
            "product_link_open_count": 4,
            "product_purchase_count": 1,
        }
    }