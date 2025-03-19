import hashlib
import random


def decode_cursor(cursor, recommendations):
    """Decode the cursor to get the offset."""
    if not cursor:
        return 0  # Start from the beginning

    try:
        last_seen_product_id = cursor.split('_')[0]  # Extract product_id
        for idx, (prod_id, _) in enumerate(recommendations):
            if prod_id == int(last_seen_product_id):  # Ensure correct comparison
                return idx + 1  # Start from the next item
    except Exception as e:
        print(f"Cursor decode error: {e}")

    return 0  # Default to beginning


def encode_cursor(last_seen_product_id, last_score):
    """Create a new cursor for the next batch."""
    cursor_hash = hashlib.sha256(str(last_score).encode()).hexdigest()[:8]
    return f"{last_seen_product_id}_{cursor_hash}"


def paginate_recommendations(recommendations, seen, cursor, limit):
    """
    Paginate recommendations while keeping track of already seen items.

    :param recommendations: List of (product_id, score) tuples.
    :param seen: Set of already seen product IDs.
    :param cursor: Encoded cursor to track pagination.
    :param limit: Number of recommendations to return per request.
    :return: Tuple (paginated_product_ids, next_cursor)
    """
    start_index = decode_cursor(cursor, recommendations)

    # Fetch the next batch of recommendations
    paginated = recommendations[start_index:start_index + limit]

    # Shuffle a portion of the top-ranked items to keep recommendations fresh
    if len(paginated) > 1:
        top_threshold = max(3, int(0.5 * len(paginated)))
        shuffled_part = paginated[:top_threshold]
        random.shuffle(shuffled_part)
        paginated[:top_threshold] = shuffled_part

    # Get product IDs, making sure to remove duplicates
    paginated_product_ids = [rec[0] for rec in paginated if rec[0] not in seen]

    # Generate next cursor if there are more products to load
    next_cursor = (
        encode_cursor(paginated[-1][0], paginated[-1][1]) if len(paginated) == limit else None
    )

    return paginated_product_ids, next_cursor
