import pytest

from src.discovery_logic import build_catalog_request, normalize_catalog_response


def test_builds_natural_language_global_catalog_request():
    request = build_catalog_request(
        "A light rain jacket for hiking under $100",
        country="CA",
        currency="CAD",
    )

    assert request["method"] == "tools/call"
    assert request["params"]["name"] == "search_catalog"
    arguments = request["params"]["arguments"]
    assert arguments["catalog"]["query"] == "A light rain jacket for hiking under $100"
    assert arguments["catalog"]["context"]["address_country"] == "CA"
    assert arguments["catalog"]["context"]["currency"] == "CAD"
    assert arguments["catalog"]["filters"]["available"] is True
    assert arguments["meta"]["ucp-agent"]["profile"].startswith("https://")


def test_applies_maximum_price_as_currency_minor_units():
    request = build_catalog_request(
        "hiking jacket", currency="USD", max_price=99.95
    )

    assert request["params"]["arguments"]["catalog"]["filters"]["price"] == {
        "max": 9995
    }


def test_applies_zero_decimal_currency_price_filter():
    request = build_catalog_request(
        "coat", currency="JPY", max_price=10000
    )

    assert request["params"]["arguments"]["catalog"]["filters"]["price"] == {
        "max": 10000
    }


def test_rejects_empty_catalog_query():
    with pytest.raises(ValueError, match="cannot be empty"):
        build_catalog_request("  ")


def test_normalizes_mcp_catalog_response_and_minor_currency_units():
    response = {
        "result": {
            "structuredContent": {
                "products": [
                    {
                        "id": "gid://shopify/p/example",
                        "title": "Trail Shoe",
                        "description": {"plain": "Comfortable for long runs"},
                        "url": "https://shop.example/products/trail-shoe",
                        "price_range": {"min": {"amount": 12000, "currency": "USD"}},
                        "media": [
                            {"type": "image", "url": "https://cdn.example/shoe.jpg"}
                        ],
                        "variants": [
                            {
                                "price": {"amount": 12999, "currency": "USD"},
                                "availability": {"available": True},
                                "seller": {
                                    "name": "Example Running",
                                    "domain": "shop.example",
                                },
                            }
                        ],
                    }
                ]
            }
        }
    }

    products = normalize_catalog_response(response)

    assert products == [
        {
            "title": "Trail Shoe",
            "price": "$129.99",
            "brand": "Example Running",
            "availability": "In Stock",
            "description": "Comfortable for long runs",
            "url": "https://shop.example/products/trail-shoe",
            "image_url": "https://cdn.example/shoe.jpg",
        }
    ]


def test_skips_unavailable_or_unsafe_product_links():
    response = {
        "result": {
            "structuredContent": {
                "products": [
                    {
                        "title": "Nope",
                        "url": "javascript:alert(1)",
                        "variants": [
                            {
                                "availability": {"available": False},
                                "seller": {"name": "Bad", "domain": "bad.example"},
                            }
                        ],
                    }
                ]
            }
        }
    }

    assert normalize_catalog_response(response) == []