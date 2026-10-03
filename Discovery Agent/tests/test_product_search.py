from src import product_search


class FakeResponse:
    def __init__(self, payload=None, content_type="application/json", text=""):
        self._payload = payload
        self.headers = {"Content-Type": content_type}
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_decodes_streamable_http_sse_response():
    response = FakeResponse(
        content_type="text/event-stream",
        text=(
            'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}'
            '\n\nevent: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"structuredContent":{"products":[]}}}\n\n'
        ),
    )

    assert product_search._decode_mcp_response(response) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"structuredContent": {"products": []}},
    }


def test_search_calls_global_catalog_once_and_normalizes_results(monkeypatch):
    response_payload = {
        "result": {
            "structuredContent": {
                "products": [
                    {
                        "id": "gid://shopify/p/example",
                        "title": "Warm Hiking Jacket",
                        "url": "https://shop.example/products/jacket",
                        "price_range": {"min": {"amount": 8500, "currency": "USD"}},
                        "variants": [
                            {
                                "price": {"amount": 8500, "currency": "USD"},
                                "availability": {"available": True},
                                "seller": {
                                    "name": "Example Outdoors",
                                    "domain": "shop.example",
                                },
                            }
                        ],
                    }
                ]
            }
        }
    }
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(response_payload)

    monkeypatch.setattr(product_search.requests, "post", fake_post)

    products = product_search.search_shopify_dynamic(
        "a warm jacket for hiking", max_price=100
    )

    assert len(calls) == 1
    url, request = calls[0]
    assert url == product_search.GLOBAL_CATALOG_MCP_URL
    arguments = request["json"]["params"]["arguments"]["catalog"]
    assert arguments["query"] == "a warm jacket for hiking"
    assert arguments["filters"]["price"]["max"] == 10000
    assert products[0]["title"] == "Warm Hiking Jacket"
    assert products[0]["price"] == "$85.00"


def test_aggregates_configured_marketplaces_fairly(monkeypatch):
    monkeypatch.setenv("EBAY_CLIENT_ID", "client")
    monkeypatch.setenv("EBAY_CLIENT_SECRET", "secret")
    monkeypatch.setenv("AMAZON_CLIENT_ID", "client")
    monkeypatch.setenv("AMAZON_CLIENT_SECRET", "secret")
    monkeypatch.setenv("AMAZON_PARTNER_TAG", "partner-20")
    monkeypatch.setattr(
        product_search,
        "search_shopify_catalog",
        lambda *_: [{"title": f"Shopify {index}"} for index in range(4)],
    )
    monkeypatch.setattr(
        product_search,
        "search_ebay",
        lambda *_: [{"title": f"eBay {index}"} for index in range(3)],
    )
    monkeypatch.setattr(
        product_search,
        "search_amazon",
        lambda *_: [{"title": f"Amazon {index}"} for index in range(2)],
    )

    products = product_search.search_shopify_dynamic("running jacket")

    assert [product["title"] for product in products[:6]] == [
        "Shopify 0",
        "eBay 0",
        "Amazon 0",
        "Shopify 1",
        "eBay 1",
        "Amazon 1",
    ]


def test_llm_selected_provider_limits_search_to_that_provider(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "catalog-project")
    monkeypatch.setattr(
        product_search,
        "search_shopify_catalog",
        lambda *_: (_ for _ in ()).throw(AssertionError("Shopify should not run")),
    )
    monkeypatch.setattr(
        product_search,
        "search_google",
        lambda *_, **__: [{"title": "Owned catalog item", "source": "Google"}],
    )

    products = product_search.search_shopify_dynamic(
        "product in our catalog", providers=["Google"]
    )

    assert products == [{"title": "Owned catalog item", "source": "Google"}]


def test_google_search_is_only_enabled_with_project_configured(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert product_search.get_enabled_providers() == ["Shopify"]

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "retail-catalog-project")
    assert product_search.get_enabled_providers() == ["Shopify", "Google"]


def test_llm_search_tool_requires_enabled_provider_selection():
    tool = product_search.build_search_tool(["Shopify", "eBay", "Google"])
    parameters = tool["function"]["parameters"]

    assert parameters["required"] == ["prompt", "providers"]
    assert parameters["properties"]["providers"]["items"]["enum"] == [
        "Shopify",
        "eBay",
        "Google",
    ]


def test_normalizes_google_owned_catalog_results_and_budget():
    payload = {
        "results": [
            {
                "product": {
                    "title": "Organic cotton sweater",
                    "uri": "https://store.example/products/sweater",
                    "description": "A soft sweater from our inventory.",
                    "brands": ["Example Brand"],
                    "availability": "IN_STOCK",
                    "priceInfo": {"price": 72, "currencyCode": "USD"},
                    "images": [{"uri": "https://cdn.example/sweater.jpg"}],
                }
            },
            {
                "product": {
                    "title": "Expensive sweater",
                    "uri": "https://store.example/products/expensive",
                    "priceInfo": {"price": 220, "currencyCode": "USD"},
                }
            },
        ]
    }

    products = product_search._normalize_google_results(payload, max_price=100)

    assert products == [
        {
            "title": "Organic cotton sweater",
            "price": "USD 72",
            "brand": "Example Brand · Google",
            "availability": "In Stock",
            "description": "A soft sweater from our inventory.",
            "url": "https://store.example/products/sweater",
            "image_url": "https://cdn.example/sweater.jpg",
            "source": "Google",
        }
    ]


def test_builds_google_search_request_with_budget_and_session_visitor():
    url, body = product_search._build_google_search_request(
        project="retail-project",
        location="global",
        catalog="default_catalog",
        serving_config="default_search",
        prompt="warm winter coat",
        max_price=125.5,
        visitor_id="anonymous-session-id",
    )

    assert url == (
        "https://retail.googleapis.com/v2/projects/retail-project/locations/global/"
        "catalogs/default_catalog/servingConfigs/default_search:search"
    )
    assert body == {
        "query": "warm winter coat",
        "pageSize": 6,
        "visitorId": "anonymous-session-id",
        "filter": 'availability: ANY("IN_STOCK") AND price: IN(*, 125.5i)',
    }


def test_gets_and_caches_ebay_application_token(monkeypatch):
    product_search._token_cache.clear()
    monkeypatch.setenv("EBAY_CLIENT_ID", "test-client")
    monkeypatch.setenv("EBAY_CLIENT_SECRET", "test-secret")
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse({"access_token": "ebay-token", "expires_in": 7200})

    monkeypatch.setattr(product_search.requests, "post", fake_post)

    assert product_search._get_ebay_token() == "ebay-token"
    assert product_search._get_ebay_token() == "ebay-token"
    assert len(calls) == 1
    assert calls[0][1]["data"]["grant_type"] == "client_credentials"


def test_searches_ebay_browse_api_and_normalizes_listing(monkeypatch):
    monkeypatch.setenv("EBAY_MARKETPLACE_ID", "EBAY_US")
    monkeypatch.setattr(product_search, "_get_ebay_token", lambda: "ebay-token")

    def fake_get(url, **kwargs):
        assert url == product_search.EBAY_SEARCH_URL
        assert kwargs["params"]["q"] == "waterproof hiking boots"
        assert kwargs["headers"]["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_US"
        return FakeResponse(
            {
                "itemSummaries": [
                    {
                        "title": "Waterproof Hiking Boots",
                        "itemWebUrl": "https://www.ebay.com/itm/123",
                        "price": {"value": "89.99", "currency": "USD"},
                        "seller": {"username": "outdoor-seller"},
                        "condition": "New",
                        "image": {"imageUrl": "https://i.ebayimg.com/boots.jpg"},
                        "estimatedAvailabilities": [
                            {"estimatedAvailabilityStatus": "IN_STOCK"}
                        ],
                    },
                    {
                        "title": "Too expensive boots",
                        "itemWebUrl": "https://www.ebay.com/itm/456",
                        "price": {"value": "180.00", "currency": "USD"},
                    },
                ]
            }
        )

    monkeypatch.setattr(product_search.requests, "get", fake_get)

    results = product_search.search_ebay(
        "waterproof hiking boots", max_price=100
    )

    assert len(results) == 1
    assert results[0]["title"] == "Waterproof Hiking Boots"
    assert results[0]["brand"] == "outdoor-seller · eBay"
    assert results[0]["price"] == "USD 89.99"
    assert results[0]["source"] == "eBay"


def test_searches_amazon_creators_api_and_normalizes_item(monkeypatch):
    product_search._token_cache.clear()
    monkeypatch.setenv("AMAZON_CLIENT_ID", "test-client")
    monkeypatch.setenv("AMAZON_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("AMAZON_PARTNER_TAG", "partner-20")
    monkeypatch.setattr(product_search, "_get_amazon_token", lambda: "amazon-token")

    def fake_post(url, **kwargs):
        assert url == product_search.AMAZON_SEARCH_URL
        assert kwargs["headers"]["Authorization"] == "Bearer amazon-token"
        assert kwargs["json"]["keywords"] == "wireless headphones"
        assert kwargs["json"]["partnerTag"] == "partner-20"
        return FakeResponse(
            {
                "searchResult": {
                    "items": [
                        {
                            "detailPageURL": "https://www.amazon.com/dp/ASIN123?tag=partner-20",
                            "itemInfo": {
                                "title": {"displayValue": "Wireless Headphones"},
                                "byLineInfo": {
                                    "brand": {"displayValue": "Audio Brand"}
                                },
                                "features": {"displayValues": ["Noise cancelling"]},
                            },
                            "images": {
                                "primary": {
                                    "large": {"url": "https://images.amazon.com/headphones.jpg"}
                                }
                            },
                            "offersV2": {
                                "listings": [
                                    {"price": {"money": {"amount": "79.99", "currency": "USD"}}}
                                ]
                            },
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(product_search.requests, "post", fake_post)

    results = product_search.search_amazon("wireless headphones", max_price=100)

    assert results == [
        {
            "title": "Wireless Headphones",
            "price": "USD 79.99",
            "brand": "Audio Brand · Amazon",
            "availability": "See Amazon for availability",
            "description": "Noise cancelling",
            "url": "https://www.amazon.com/dp/ASIN123?tag=partner-20",
            "image_url": "https://images.amazon.com/headphones.jpg",
            "source": "Amazon",
        }
    ]
