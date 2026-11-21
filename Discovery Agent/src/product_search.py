"""Shopify Global Catalog MCP client for product discovery."""

import json
import os

import requests

from .discovery_logic import build_catalog_request, normalize_catalog_response


GLOBAL_CATALOG_MCP_URL = "https://catalog.shopify.com/api/ucp/mcp"


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


def build_search_tool() -> dict:
    """Build the Shopify-only tool schema for the shopping assistant."""
    return {
        "type": "function",
        "function": {
            "name": "search_shopify_catalog",
            "description": "Search Shopify's Global Catalog for matching products.",
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
                            "Optional maximum price in the configured Shopify catalog currency, "
                            "only when the shopper explicitly stated a budget cap."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        },
    }


def search_shopify_catalog(prompt: str = "", max_price: float = None) -> list:
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
    return [_with_shopify_source(product) for product in products]


def _with_shopify_source(product: dict) -> dict:
    """Copy a normalized product card with its Shopify source label."""
    item = dict(product)
    item["source"] = "Shopify"
    if item.get("brand"):
        item["brand"] = f"{item['brand']} · Shopify"
    else:
        item["brand"] = "Shopify"
    return item
