"""Deterministic discovery rules, kept independent from the Streamlit UI."""

import re


POPULAR_SHOPIFY_STORES = [
    "gymshark.com",
    "allbirds.com",
    "kith.com",
    "fabletics.com",
    "aloyoga.com",
    "chubbieshorts.com",
    "rothys.com",
    "taylormade-golf.com",
    "cotopaxi.com",
    "vuoriclothing.com",
    "rhone.com",
    "marine-layer.com",
    "mottandbow.com",
    "bombas.com",
    "noble-apparel.com",
    "unTuckit.com",
    "brooklinen.com",
    "outerknown.com",
    "tentree.com",
    "blundstone.com",
    "tuckernuck.com",
]

CATEGORY_KEYWORDS = {
    "t-shirt": ["t-shirt", "t-shirts", "tshirt", "tshirts", "tee", "tees"],
    "towel": ["towel", "towels", "washcloth"],
    "slippers": ["slipper", "slippers", "slides", "sandals"],
    "shorts": ["shorts", "short"],
    "pants": ["pants", "trousers", "jeans", "joggers", "leggings"],
    "shoes": ["shoes", "sneakers", "boots", "footwear"],
    "jacket": ["jacket", "coat", "outerwear", "parka"],
    "gymwear": ["gymwear", "activewear", "workout", "fitness", "gym", "sportswear"],
}


def parse_user_prompt(user_prompt: str):
    """Extract query, price, gender, brand, category, and external-brand flag."""
    query = user_prompt
    max_price = None
    gender = None
    target_brand = None
    category = None
    is_external_brand = False

    price_match = re.search(r"under\s*\$?(\d+(?:\.\d+)?)", user_prompt, re.IGNORECASE)
    if price_match:
        max_price = float(price_match.group(1))
        query = re.sub(r"under\s*\$?(\d+(?:\.\d+)?)", "", query, flags=re.IGNORECASE).strip()

    if re.search(r"\b(women|womens|female|woman|ladies)\b", user_prompt, re.IGNORECASE):
        gender = "female"
    elif re.search(r"\b(men|mens|male|man|guys)\b", user_prompt, re.IGNORECASE):
        gender = "male"

    for cat_key, synonyms in CATEGORY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(syn)}\b", user_prompt, re.IGNORECASE) for syn in synonyms):
            category = cat_key
            break

    for store in POPULAR_SHOPIFY_STORES:
        brand_name = store.split(".")[0].lower()
        if re.search(rf"\b{re.escape(brand_name)}\b", user_prompt, re.IGNORECASE):
            target_brand = store
            break

    if not target_brand:
        explicit_brand_match = re.search(
            r"\b(brand\s+([a-zA-Z0-9]+)|([a-zA-Z0-9]+)\s+brand)\b",
            user_prompt,
            re.IGNORECASE,
        )
        if explicit_brand_match:
            extracted_brand = explicit_brand_match.group(2) or explicit_brand_match.group(3)
            target_brand = extracted_brand.lower()
            is_external_brand = True
        else:
            known_external_brands = [
                "nike", "adidas", "puma", "under armour", "reebok", "zara", "h&m", "uniqlo"
            ]
            for external_brand in known_external_brands:
                if re.search(rf"\b{re.escape(external_brand)}\b", user_prompt, re.IGNORECASE):
                    target_brand = external_brand
                    is_external_brand = True
                    break

    return query, max_price, gender, target_brand, category, is_external_brand
