"""Gemini text embeddings and candidate-level cosine similarity."""

from __future__ import annotations

import hashlib
import math

import requests

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{EMBEDDING_MODEL}:batchEmbedContents"
)


class EmbeddingError(RuntimeError):
    """Raised when Gemini cannot generate valid retrieval embeddings."""


def _product_text(product: dict) -> str:
    return "\n".join(
        (
            f"title: {product.get('title', '')}",
            f"brand: {product.get('brand', '')}",
            f"description: {product.get('description', '')}",
        )
    )[:12000]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two non-empty vectors."""
    if not left or len(left) != len(right):
        raise ValueError("Embedding vectors must have the same non-zero length.")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Embedding vectors must contain only finite numbers.")

    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if not left_norm or not right_norm:
        raise ValueError("Embedding vectors must have a non-zero norm.")

    score = math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))


def embed_search_candidates(
    query: str, products: list[dict], api_key: str
) -> tuple[list[float], list[dict]]:
    """Embed a query and its provider-returned products, ranked by cosine score."""
    if not api_key.strip():
        raise EmbeddingError("GEMINI_API_KEY is required to generate embeddings.")

    product_texts = [_product_text(product) for product in products]
    request_items = [
        {
            "model": f"models/{EMBEDDING_MODEL}",
            "content": {"parts": [{"text": query[:12000]}]},
            "taskType": "RETRIEVAL_QUERY",
        }
    ]
    request_items.extend(
        {
            "model": f"models/{EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
        }
        for text in product_texts
    )

    try:
        response = requests.post(
            EMBEDDING_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json={"requests": request_items},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        status = getattr(error.response, "status_code", None)
        detail = f" (HTTP {status})" if status else ""
        raise EmbeddingError(f"Gemini embedding request failed{detail}.") from error

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(request_items):
        raise EmbeddingError("Gemini returned an unexpected number of embeddings.")

    vectors = []
    for item in embeddings:
        values = item.get("values") if isinstance(item, dict) else None
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in values
            )
        ):
            raise EmbeddingError("Gemini returned an invalid embedding vector.")
        vectors.append([float(value) for value in values])

    query_vector = vectors[0]
    matches = []
    for index, (product, text) in enumerate(zip(products, product_texts), start=1):
        try:
            score = cosine_similarity(query_vector, vectors[index])
        except ValueError as error:
            raise EmbeddingError("Gemini returned incompatible embedding vectors.") from error
        source = str(product.get("source", "unknown"))
        product_url = str(product.get("url", ""))
        product_key = hashlib.sha256(
            f"{source}\0{product_url}\0{product.get('title', '')}".encode("utf-8")
        ).hexdigest()
        ranked_product = {**product, "similarity_score": score}
        matches.append(
            {
                "product": ranked_product,
                "item_key": product_key,
                "text": text,
                "embedding": vectors[index],
                "score": score,
            }
        )

    matches.sort(key=lambda match: match["score"], reverse=True)
    for rank, match in enumerate(matches, start=1):
        match["rank"] = rank
    return query_vector, matches
