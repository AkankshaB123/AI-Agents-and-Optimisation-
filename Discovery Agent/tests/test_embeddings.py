import unittest
from unittest.mock import Mock, patch

from src.embeddings import EMBEDDING_MODEL, embed_search_candidates


class EmbeddingTests(unittest.TestCase):
    def test_embeds_query_and_candidates_then_ranks_by_cosine_similarity(self):
        response = Mock()
        response.json.return_value = {
            "embeddings": [
                {"values": [1.0, 0.0]},
                {"values": [0.8, 0.6]},
                {"values": [0.0, 1.0]},
            ]
        }
        products = [
            {
                "title": "Rain jacket",
                "description": "Waterproof shell",
                "brand": "Outdoor shop",
                "source": "Shopify",
                "url": "https://shop.example/jacket",
            },
            {
                "title": "Trail shoe",
                "description": "Running shoe",
                "brand": "Shoe shop",
                "source": "Shopify",
                "url": "https://shop.example/shoe",
            },
        ]

        with patch("src.embeddings.requests.post", return_value=response) as post:
            query_embedding, matches = embed_search_candidates(
                "waterproof hiking jacket", products, "test-key"
            )

        self.assertEqual(query_embedding, [1.0, 0.0])
        self.assertEqual(
            [match["product"]["title"] for match in matches],
            ["Rain jacket", "Trail shoe"],
        )
        self.assertEqual([match["rank"] for match in matches], [1, 2])
        self.assertEqual([match["score"] for match in matches], [0.8, 0.0])
        self.assertEqual(
            [match["product"]["similarity_score"] for match in matches],
            [0.8, 0.0],
        )

        request_body = post.call_args.kwargs["json"]
        self.assertEqual(len(request_body["requests"]), 3)
        self.assertEqual(
            request_body["requests"][0]["model"], f"models/{EMBEDDING_MODEL}"
        )
        self.assertEqual(request_body["requests"][0]["taskType"], "RETRIEVAL_QUERY")
        self.assertEqual(
            request_body["requests"][1]["taskType"], "RETRIEVAL_DOCUMENT"
        )
