"""Shopify Global Catalog MCP client for product discovery."""

import base64
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

import requests

from .discovery_logic import build_catalog_request, normalize_catalog_response


GLOBAL_CATALOG_MCP_URL = "https://catalog.shopify.com/api/ucp/mcp"
EBAY_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
AMAZON_TOKEN_ENDPOINTS = {
    "NA": "https://api.amazon.com/auth/o2/token",
    "EU": "https://api.amazon.co.uk/auth/o2/token",
    "FE": "https://api.amazon.co.jp/auth/o2/token",
}
AMAZON_SEARCH_URL = "https://creatorsapi.amazon/catalog/v1/searchItems"
GOOGLE_RETAIL_API_ROOT = "https://retail.googleapis.com/v2"
_token_cache = {}


def _setting(name: str, default: str = "") -> str:
    """Read a setting from environment variables or flat Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


def _decode_mcp_response(response: requests.Response) -> dict:
    """Decode JSON or an SSE stream and return the final JSON-RPC message."""
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/event-stream" not in content_type:
        return response.json()

    messages = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            messages.append(json.loads(data))
        except json.JSONDecodeError:
            continue

    final_message = next(
        (message for message in reversed(messages) if "result" in message or "error" in message),
        None,
    )
    if final_message is None:
        raise ValueError("Shopify Catalog MCP returned no JSON-RPC result.")
    return final_message


def get_enabled_providers() -> list:
    """Return providers configured for this deployment; Shopify is always enabled."""
    providers = ["Shopify"]
    if _setting("EBAY_CLIENT_ID") and _setting("EBAY_CLIENT_SECRET"):
        providers.append("eBay")
    if all(
        _setting(name)
        for name in ("AMAZON_CLIENT_ID", "AMAZON_CLIENT_SECRET", "AMAZON_PARTNER_TAG")
    ):
        providers.append("Amazon")
    if _setting("GOOGLE_CLOUD_PROJECT"):
        providers.append("Google")
    return providers


def build_search_tool(enabled_providers: list) -> dict:
    """Build the LLM function schema from only the providers enabled in config."""
    return {
        "type": "function",
        "function": {
            "name": "search_shopify_dynamic",
            "description": (
                "Searches selected connected product catalogs using the shopper's "
                "natural-language request. Choose provider(s) based on query intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The shopper's natural-language product request, preserving "
                            "preferences such as budget, intended use, style, gender, or brand."
                        ),
                    },
                    "max_price": {
                        "type": "number",
                        "description": (
                            "Optional maximum price in the configured catalog currency, "
                            "only when the shopper explicitly stated a budget cap."
                        ),
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string", "enum": enabled_providers},
                        "description": (
                            "Provider(s) selected based on query intent. Explicit eBay/Amazon "
                            "requests should prefer that marketplace; Google is an owned "
                            "catalog, not public shopping search."
                        ),
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
                "required": ["prompt", "providers"],
            },
        },
    }


def search_shopify_dynamic(
    prompt: str = "",
    max_price: float = None,
    providers: list = None,
    visitor_id: str = None,
) -> list:
    """Search LLM-selected configured catalogs and interleave normalized offers."""
    enabled_providers = get_enabled_providers()
    selected = [
        provider
        for provider in (providers or enabled_providers)
        if provider in enabled_providers
    ]
    if not selected:
        raise ValueError("Select at least one configured product catalog.")

    search_functions = {
        "Shopify": lambda: search_shopify_catalog(prompt, max_price),
        "eBay": lambda: search_ebay(prompt, max_price),
        "Amazon": lambda: search_amazon(prompt, max_price),
        "Google": lambda: search_google(prompt, max_price, visitor_id=visitor_id),
    }
    selected_searches = [
        (provider, search_functions[provider]) for provider in selected
    ]

    results_by_provider = {}
    failures = []
    with ThreadPoolExecutor(max_workers=len(selected_searches)) as executor:
        futures = {
            executor.submit(search_fn): provider
            for provider, search_fn in selected_searches
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                results_by_provider[provider] = future.result()
            except Exception as error:
                failures.append(f"{provider}: {error}")

    ordered_results = {
        provider: results_by_provider[provider]
        for provider, _ in selected_searches
        if provider in results_by_provider
    }
    products = _interleave_results(ordered_results, limit=12)
    if products:
        return products
    if failures:
        raise RuntimeError("All configured product searches failed. " + "; ".join(failures))
    return []


def search_google(
    prompt: str,
    max_price: float = None,
    limit: int = 6,
    visitor_id: str = None,
) -> list:
    """Search a deployment-owned Vertex AI Search for commerce product catalog."""
    project = _setting("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured.")

    try:
        import google.auth
        from google.oauth2 import service_account
        from google.auth.transport.requests import AuthorizedSession
    except ImportError as error:
        raise RuntimeError(
            "Google search dependencies are missing; install google-auth."
        ) from error

    location = _setting("GOOGLE_CLOUD_LOCATION", "global")
    catalog = _setting("GOOGLE_CLOUD_CATALOG_ID", "default_catalog")
    serving_config = _setting("GOOGLE_CLOUD_SERVING_CONFIG", "default_search")
    url, body = _build_google_search_request(
        project=project,
        location=location,
        catalog=catalog,
        serving_config=serving_config,
        prompt=prompt,
        max_price=max_price,
        limit=limit,
        visitor_id=visitor_id or uuid.uuid4().hex,
    )

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    service_account_info = _setting("GOOGLE_SERVICE_ACCOUNT_INFO")
    if service_account_info:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(service_account_info), scopes=scopes
        )
    else:
        credentials, _ = google.auth.default(scopes=scopes)
    response = AuthorizedSession(credentials).post(url, json=body, timeout=20)
    response.raise_for_status()
    return _normalize_google_results(
        response.json(), limit=limit, max_price=max_price
    )


def _build_google_search_request(
    project: str,
    location: str,
    catalog: str,
    serving_config: str,
    prompt: str,
    max_price: float = None,
    limit: int = 6,
    visitor_id: str = "",
) -> tuple:
    """Build the documented Vertex AI Search for commerce request."""
    serving_config_path = (
        f"projects/{project}/locations/{location}/catalogs/{catalog}/"
        f"servingConfigs/{serving_config}"
    )
    url = f"{GOOGLE_RETAIL_API_ROOT}/{serving_config_path}:search"
    body = {
        "query": prompt,
        "pageSize": min(limit, 24),
        "visitorId": visitor_id or uuid.uuid4().hex,
        "filter": 'availability: ANY("IN_STOCK")',
    }
    if max_price is not None:
        body["filter"] += f" AND price: IN(*, {float(max_price)}i)"
    return url, body


def _normalize_google_results(
    payload: dict, limit: int = 6, max_price: float = None
) -> list:
    """Convert Vertex AI Search product results to shared product-card objects."""
    results = []
    for result in payload.get("results", []):
        product = result.get("product") or {}
        product_url = product.get("uri", "")
        if not product_url.startswith("https://"):
            continue

        price_info = product.get("priceInfo", {})
        price = price_info.get("price")
        currency = price_info.get("currencyCode")
        numeric_price = _parse_amount(price)
        if max_price is not None and numeric_price is not None and numeric_price > max_price:
            continue
        availability = product.get("availability", "")
        if availability == "OUT_OF_STOCK":
            continue
        availability_text = {
            "IN_STOCK": "In Stock",
            "OUT_OF_STOCK": "Out of Stock",
            "PREORDER": "Pre-order",
            "BACKORDER": "Back-order",
        }.get(availability, "Availability varies")
        images = product.get("images", [])
        image_url = images[0].get("uri", "") if images else ""
        description = product.get("description", "")
        brands = product.get("brands", [])

        results.append(
            _with_source(
                {
                    "title": product.get("title", "Google catalog product"),
                    "price": (
                        f"{currency} {price}"
                        if price is not None and currency
                        else "Price unavailable"
                    ),
                    "brand": brands[0] if brands else "Retail catalog",
                    "availability": availability_text,
                    "description": str(description)[:180],
                    "url": product_url,
                    "image_url": image_url,
                },
                "Google",
            )
        )
        if len(results) >= limit:
            break

    return results


def search_shopify_catalog(prompt: str, max_price: float = None) -> list:
    """Search Shopify merchants with one natural-language Global Catalog MCP call."""
    profile_url = _setting("SHOPIFY_UCP_AGENT_PROFILE").strip()
    country = _setting("SHOPIFY_CATALOG_COUNTRY", "US").strip().upper()
    currency = _setting("SHOPIFY_CATALOG_CURRENCY", "USD").strip().upper()
    request_body = build_catalog_request(
        prompt=prompt,
        profile_url=profile_url or None,
        country=country,
        currency=currency,
        max_price=max_price,
    )

    try:
        response = requests.post(
            GLOBAL_CATALOG_MCP_URL,
            json=request_body,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = _decode_mcp_response(response)
    except requests.RequestException as error:
        raise RuntimeError(f"Shopify Catalog MCP request failed: {error}") from error
    except ValueError as error:
        raise RuntimeError("Shopify Catalog MCP returned invalid JSON.") from error

    products = normalize_catalog_response(payload, limit=12)
    return [_with_source(product, "Shopify") for product in products]


def _with_source(product: dict, source: str) -> dict:
    """Copy a normalized product card with its marketplace source label."""
    item = dict(product)
    item["source"] = source
    if item.get("brand"):
        item["brand"] = f"{item['brand']} · {source}"
    else:
        item["brand"] = source
    return item


def _interleave_results(results_by_provider: dict, limit: int) -> list:
    """Round-robin marketplace results so one provider cannot crowd out others."""
    providers = list(results_by_provider)
    mixed = []
    index = 0
    while len(mixed) < limit:
        added = False
        for provider in providers:
            provider_results = results_by_provider[provider]
            if index < len(provider_results):
                mixed.append(provider_results[index])
                added = True
                if len(mixed) >= limit:
                    break
        if not added:
            break
        index += 1
    return mixed


def _get_ebay_token() -> str:
    """Mint and reuse an eBay Browse API application token."""
    client_id = _setting("EBAY_CLIENT_ID")
    client_secret = _setting("EBAY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("eBay Browse API credentials are not configured.")
    cache_key = "ebay"
    cached = _token_cache.get(cache_key)
    if cached and cached[1] > time.time():
        return cached[0]

    credentials = base64.b64encode(
        f"{client_id}:{client_secret}".encode("utf-8")
    ).decode("ascii")
    token_response = requests.post(
        EBAY_TOKEN_URL,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        },
        timeout=15,
    )
    token_response.raise_for_status()
    token_data = token_response.json()
    token = token_data["access_token"]
    _token_cache[cache_key] = (token, time.time() + token_data.get("expires_in", 7200) - 60)
    return token


def search_ebay(prompt: str, max_price: float = None, limit: int = 6) -> list:
    """Search eBay Browse API and normalize listing summaries."""
    marketplace = _setting("EBAY_MARKETPLACE_ID", "EBAY_US")
    response = requests.get(
        EBAY_SEARCH_URL,
        params={"q": prompt, "limit": min(limit, 50), "sort": "BEST_MATCH"},
        headers={
            "Authorization": f"Bearer {_get_ebay_token()}",
            "X-EBAY-C-MARKETPLACE-ID": marketplace,
        },
        timeout=20,
    )
    response.raise_for_status()
    listings = response.json().get("itemSummaries", [])
    results = []
    for listing in listings:
        price_info = listing.get("price", {})
        price_value = _parse_amount(price_info.get("value"))
        if max_price is not None and price_value is not None and price_value > max_price:
            continue
        product_url = listing.get("itemWebUrl", "")
        if not product_url.startswith("https://"):
            continue
        results.append(
            _with_source(
                {
                    "title": listing.get("title", "eBay listing"),
                    "price": _format_market_price(price_info),
                    "brand": listing.get("seller", {}).get("username", "eBay seller"),
                    "availability": (
                        listing.get("estimatedAvailabilities") or [{}]
                    )[0].get("estimatedAvailabilityStatus", "Availability varies"),
                    "description": listing.get("condition", "eBay marketplace listing"),
                    "url": product_url,
                    "image_url": listing.get("image", {}).get("imageUrl", ""),
                },
                "eBay",
            )
        )
        if len(results) >= limit:
            break
    return results


def _get_amazon_token() -> str:
    """Mint and reuse an Amazon Creators API OAuth token."""
    client_id = _setting("AMAZON_CLIENT_ID")
    client_secret = _setting("AMAZON_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Amazon Creators API credentials are not configured.")
    region = _setting("AMAZON_CREDENTIAL_REGION", "NA").upper()
    if region not in AMAZON_TOKEN_ENDPOINTS:
        raise ValueError("AMAZON_CREDENTIAL_REGION must be NA, EU, or FE.")
    cache_key = f"amazon:{region}:{client_id}"
    cached = _token_cache.get(cache_key)
    if cached and cached[1] > time.time():
        return cached[0]

    response = requests.post(
        AMAZON_TOKEN_ENDPOINTS[region],
        json={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "creatorsapi::default",
        },
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    token_data = response.json()
    token = token_data["access_token"]
    _token_cache[cache_key] = (token, time.time() + token_data.get("expires_in", 3600) - 60)
    return token


def search_amazon(prompt: str, max_price: float = None, limit: int = 6) -> list:
    """Search Amazon Creators API for approved Associates partners."""
    marketplace = _setting("AMAZON_MARKETPLACE", "www.amazon.com")
    partner_tag = _setting("AMAZON_PARTNER_TAG")
    if not partner_tag:
        raise RuntimeError("Amazon Associates partner tag is not configured.")
    payload = {
        "keywords": prompt,
        "marketplace": marketplace,
        "partnerTag": partner_tag,
        "partnerType": "Associates",
        "itemCount": min(limit, 10),
        "resources": [
            "images.primary.large",
            "itemInfo.title",
            "itemInfo.features",
            "itemInfo.byLineInfo",
            "offersV2.listings.price",
        ],
    }
    response = requests.post(
        AMAZON_SEARCH_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {_get_amazon_token()}",
            "Content-Type": "application/json",
            "x-marketplace": marketplace,
        },
        timeout=20,
    )
    response.raise_for_status()
    items = response.json().get("searchResult", {}).get("items", [])
    results = []
    for item in items:
        item_info = item.get("itemInfo", {})
        title = item_info.get("title", {}).get("displayValue", "Amazon product")
        url = item.get("detailPageURL", "")
        if not url.startswith("https://"):
            continue
        price_info = (
            item.get("offersV2", {})
            .get("listings", [{}])[0]
            .get("price", {})
            .get("money", {})
        )
        amount = _parse_amount(price_info.get("amount"))
        if max_price is not None and amount is not None and amount > max_price:
            continue
        features = item_info.get("features", {}).get("displayValues", [])
        seller = item_info.get("byLineInfo", {}).get("brand", {}).get("displayValue", "Amazon")
        image_url = (
            item.get("images", {})
            .get("primary", {})
            .get("large", {})
            .get("url", "")
        )
        results.append(
            _with_source(
                {
                    "title": title,
                    "price": _format_market_price(price_info),
                    "brand": seller,
                    "availability": "See Amazon for availability",
                    "description": " ".join(features)[:180],
                    "url": url,
                    "image_url": image_url,
                },
                "Amazon",
            )
        )
        if len(results) >= limit:
            break
    return results


def _parse_amount(value) -> float:
    """Parse a marketplace amount string/number to a decimal float if present."""
    if value is None:
        return None
    try:
        return float(Decimal(str(value)))
    except Exception:
        return None


def _format_market_price(price: dict) -> str:
    """Format provider price value and currency without assuming USD."""
    amount = price.get("value") or price.get("amount")
    currency = price.get("currency") or price.get("currencyCode")
    if amount is None:
        return "Price unavailable"
    if currency:
        return f"{currency} {amount}"
    return str(amount)
