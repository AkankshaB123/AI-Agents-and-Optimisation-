"""Shopify Global Catalog MCP client for product discovery."""

import json
import os

import requests

from .discovery_logic import build_catalog_request, normalize_catalog_response


GLOBAL_CATALOG_MCP_URL = "https://catalog.shopify.com/api/ucp/mcp"


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


def search_shopify_dynamic(prompt: str = "", max_price: float = None) -> list:
    """Search Shopify merchants with one natural-language Global Catalog MCP call."""
    profile_url = os.getenv(
        "SHOPIFY_UCP_AGENT_PROFILE", ""
    ).strip()
    country = os.getenv("SHOPIFY_CATALOG_COUNTRY", "US").strip().upper()
    currency = os.getenv("SHOPIFY_CATALOG_CURRENCY", "USD").strip().upper()
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

    return normalize_catalog_response(payload)
