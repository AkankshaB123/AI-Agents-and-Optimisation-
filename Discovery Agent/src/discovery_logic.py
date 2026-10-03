"""Pure helpers for Shopify Global Catalog MCP requests and responses."""

import json
from decimal import Decimal
from urllib.parse import urlsplit

from babel.numbers import format_currency, get_currency_precision


DEFAULT_AGENT_PROFILE = (
    "https://shopify.dev/ucp/agent-profiles/examples/2026-08-25/"
    "valid-with-capabilities.json"
)


def build_catalog_request(
    prompt: str,
    profile_url: str = DEFAULT_AGENT_PROFILE,
    country: str = "US",
    currency: str = "USD",
    max_price: float = None,
) -> dict:
    """Build a Global Catalog MCP tools/call request from natural-language intent."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Search prompt cannot be empty.")
    profile_url = profile_url or DEFAULT_AGENT_PROFILE

    filters = {"available": True}
    if max_price is not None:
        precision = get_currency_precision(currency)
        filters["price"] = {
            "max": int(
                (Decimal(str(max_price)) * (Decimal(10) ** precision)).to_integral_value()
            )
        }

    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_catalog",
            "arguments": {
                "meta": {"ucp-agent": {"profile": profile_url}},
                "catalog": {
                    "query": prompt,
                    "filters": filters,
                    "context": {
                        "address_country": country,
                        "currency": currency,
                        "intent": prompt,
                    },
                    "pagination": {"limit": 20},
                },
            },
        },
    }


def _safe_https_url(value: str) -> str:
    """Only return HTTPS URLs from catalog data for clickable UI attributes."""
    if not value:
        return ""
    parsed = urlsplit(value)
    return value if parsed.scheme == "https" and parsed.netloc else ""


def _format_money(money: dict) -> str:
    """Format a UCP minor-unit amount using its ISO currency precision."""
    amount = money.get("amount")
    currency = money.get("currency")
    if amount is None or not currency:
        return "Price unavailable"

    precision = get_currency_precision(currency)
    major_units = Decimal(str(amount)) / (Decimal(10) ** precision)
    return format_currency(major_units, currency, locale="en_US")


def normalize_catalog_response(response: dict, limit: int = 6) -> list:
    """Convert MCP Global Catalog structured content to product cards."""
    if response.get("error"):
        error = response["error"]
        raise RuntimeError(error.get("message", "Shopify Catalog MCP request failed."))

    result = response.get("result", {})
    if result.get("isError"):
        details = result.get("content", [])
        message = next(
            (item.get("text") for item in details if item.get("type") == "text"),
            "Shopify Catalog MCP reported a search error.",
        )
        raise RuntimeError(message)
    content = result.get("structuredContent") or {}
    if not content:
        for block in result.get("content", []):
            if block.get("type") != "text":
                continue
            try:
                parsed = json.loads(block.get("text", ""))
            except (TypeError, ValueError):
                continue
            content = parsed.get("structuredContent", parsed)
            if content:
                break

    products = []
    seen_sellers = set()
    for product in content.get("products", []):
        variants = product.get("variants", [])
        product_price = product.get("price_range", {}).get("min", {})
        media = next(
            (item for item in product.get("media", []) if item.get("type") == "image"),
            {},
        )
        description = product.get("description", {})
        if isinstance(description, dict):
            description = description.get("plain") or description.get("html") or ""
        description = " ".join(str(description).split())

        for variant in variants:
            availability = variant.get("availability", {})
            if availability.get("available") is False:
                continue

            seller = variant.get("seller") or {}
            seller_domain = seller.get("domain", "")
            seller_name = seller.get("name") or seller_domain or "Shopify merchant"
            if not seller_domain:
                continue
            unique_key = (product.get("id"), seller_domain)
            if unique_key in seen_sellers:
                continue

            price = variant.get("price") or product_price
            product_url = _safe_https_url(product.get("url")) or _safe_https_url(
                variant.get("checkout_url")
            )
            if not product_url:
                continue

            seen_sellers.add(unique_key)
            products.append(
                {
                    "title": product.get("title", "Product"),
                    "price": _format_money(price),
                    "brand": seller_name,
                    "availability": "In Stock",
                    "description": description[:180],
                    "url": product_url,
                    "image_url": _safe_https_url(media.get("url", "")),
                }
            )
            if len(products) >= limit:
                return products

    return products
