"""Product discovery orchestration and external catalog adapters."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from duckduckgo_search import DDGS

from .discovery_logic import CATEGORY_KEYWORDS, POPULAR_SHOPIFY_STORES, parse_user_prompt


_session = requests.Session()


def fetch_store_products(
    store: str,
    query: str,
    max_price: float = None,
    gender: str = None,
    category: str = None,
) -> list:
    """Fetch matching, available products from one Shopify storefront."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    api_url = (
        f"https://{store}/search/suggest.json?q={query}"
        "&resources[type]=product&resources[limit]=10"
    )
    matched_products = []
    target_keywords = [word.lower() for word in re.findall(r"\b\w{3,}\b", query)]

    try:
        response = _session.get(api_url, headers=headers, timeout=1.2)
        if response.status_code != 200:
            return matched_products

        products = (
            response.json()
            .get("resources", {})
            .get("results", {})
            .get("products", [])
        )

        for product in products:
            if not product.get("available", True):
                continue

            title = product.get("title", "")
            price = float(product.get("price", 0.0))
            title_lower = title.lower()

            if category:
                allowed_synonyms = CATEGORY_KEYWORDS.get(category, [category])
                if not any(
                    re.search(rf"\b{re.escape(synonym)}\b", title_lower)
                    for synonym in allowed_synonyms
                ):
                    continue

            if gender == "female" and any(
                marker in title_lower for marker in ["men's", "mens", "for men"]
            ):
                continue
            if gender == "male" and any(
                marker in title_lower for marker in ["women's", "womens", "for women"]
            ):
                continue

            if target_keywords and not any(keyword in title_lower for keyword in target_keywords):
                continue
            if max_price and price > max_price:
                continue

            raw_description = (
                product.get("body")
                or product.get("summary")
                or "No description provided."
            )
            clean_description = re.sub(r"<[^>]+>", "", raw_description).strip()[:100] + "..."

            image_url = product.get("image") or product.get("featured_image", {}).get("src")
            if image_url:
                if image_url.startswith("//"):
                    image_url = f"https:{image_url}"
                elif not image_url.startswith("http"):
                    image_url = f"https://{store}/{image_url.lstrip('/')}"

            raw_url = product.get("url", "")
            product_url = f"https://{store}{raw_url}" if raw_url.startswith("/") else raw_url

            matched_products.append(
                {
                    "title": title,
                    "price": f"${price:.2f}",
                    "brand": store,
                    "availability": "In Stock",
                    "description": clean_description,
                    "url": product_url,
                    "image_url": image_url,
                }
            )
    except Exception:
        pass

    return matched_products


def fetch_duckduckgo_products(query: str, brand: str = None) -> list:
    """Search web results for products and normalize them for the UI."""
    products = []
    search_term = f"buy {brand} {query} price" if brand else f"buy {query} product price"
    try:
        with DDGS() as ddgs:
            text_results = list(ddgs.text(keywords=search_term, max_results=6))
            image_results = list(
                ddgs.images(
                    keywords=f"{brand or ''} {query} product photo".strip(),
                    max_results=6,
                )
            )

            for index, result in enumerate(text_results):
                title = result.get("title", "Product")
                snippet = result.get("body", "")
                url = result.get("href", "#")
                price_match = re.search(r"\$\d+(?:\.\d{2})?", title + " " + snippet)
                price = price_match.group(0) if price_match else "Check Site"
                image_url = (
                    image_results[index].get("image")
                    if index < len(image_results)
                    else "https://via.placeholder.com/200x200?text=No+Image"
                )
                domain = (
                    brand.upper()
                    if brand
                    else url.split("/")[2].replace("www.", "")
                    if "://" in url
                    else "Web Store"
                )

                products.append(
                    {
                        "title": title,
                        "price": price,
                        "brand": domain,
                        "availability": "In Stock",
                        "description": snippet[:100] + "...",
                        "url": url,
                        "image_url": image_url,
                    }
                )
    except Exception:
        pass

    return products


def search_shopify_dynamic(prompt: str = "", user_intent: dict = None) -> list:
    """Route a prompt to direct Shopify searches or external-brand web search."""
    query, max_price, gender, target_brand, category, is_external_brand = (
        parse_user_prompt(prompt)
    )

    if user_intent is not None:
        if gender:
            user_intent["gender"] = gender
        if category:
            user_intent["category"] = category
        if target_brand:
            user_intent["brand"] = target_brand

    if is_external_brand:
        return fetch_duckduckgo_products(query=prompt, brand=target_brand)[:6]

    stores_to_search = (
        [target_brand]
        if target_brand and not is_external_brand
        else POPULAR_SHOPIFY_STORES
    )
    shopify_products = []

    with ThreadPoolExecutor(max_workers=25) as executor:
        store_futures = [
            executor.submit(
                fetch_store_products, store, query, max_price, gender, category
            )
            for store in stores_to_search
        ]
        web_future = executor.submit(fetch_duckduckgo_products, query=prompt)

        for future in as_completed(store_futures):
            result = future.result()
            if result:
                shopify_products.extend(result)
        web_products = web_future.result()

    unique_products = []
    seen_stores = set()
    for product in shopify_products + web_products:
        if product["brand"] not in seen_stores:
            seen_stores.add(product["brand"])
            unique_products.append(product)
            if len(unique_products) == 6:
                break

    return unique_products
