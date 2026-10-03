from src.discovery_logic import parse_user_prompt


def test_extracts_price_gender_and_category():
    assert parse_user_prompt("women's t-shirts under $50") == (
        "women's t-shirts",
        50.0,
        "female",
        None,
        "t-shirt",
        False,
    )


def test_routes_known_shopify_brand():
    result = parse_user_prompt("Gymshark men's gymwear")

    assert result[2:] == ("male", "gymshark.com", "gymwear", False)


def test_recognizes_external_brand():
    result = parse_user_prompt("Nike brand running shoes")

    assert result[3:] == ("nike", "shoes", True)