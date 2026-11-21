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


def test_shopify_search_calls_catalog_once_and_normalizes_results(monkeypatch):
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
    products = product_search.search_shopify_catalog(
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
    assert products[0]["source"] == "Shopify"


def test_search_tool_exposes_only_shopify_search():
    tool = product_search.build_search_tool()
    parameters = tool["function"]["parameters"]

    assert tool["function"]["name"] == "search_shopify_catalog"
    assert parameters["required"] == ["prompt"]
    assert set(parameters["properties"]) == {"prompt", "max_price"}
