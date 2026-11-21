import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DISCOVERY_DB_PATH"] = str(Path(self.tmp.name) / "test.db")
        from src import persistence
        persistence.init_db()
        self.persistence = persistence

    def tearDown(self):
        os.environ.pop("DISCOVERY_DB_PATH", None)
        self.tmp.cleanup()

    def test_message_round_trip_and_isolation(self):
        first = self.persistence.create_conversation()
        second = self.persistence.create_conversation()
        self.persistence.save_message(first, "user", "Find a rain jacket")
        self.persistence.save_message(first, "assistant", "Here are some options.")
        self.persistence.save_message(second, "user", "Find running shoes")

        messages = self.persistence.load_messages(first)
        self.assertEqual([m["content"] for m in messages], [
            "Find a rain jacket",
            "Here are some options.",
        ])
        self.assertEqual(len(self.persistence.load_messages(second)), 1)

    def test_preferences_search_and_agent_state_round_trip(self):
        conversation = self.persistence.create_conversation()
        self.persistence.save_preferences(
            conversation, {"budget": 100, "category": "rain jacket"}
        )
        self.persistence.save_search(
            conversation,
            "lightweight rain jacket under 100",
            ["Shopify"],
            [{"title": "Jacket", "price": "$80"}],
        )
        self.persistence.save_agent_state(
            conversation, {"last_query": "lightweight rain jacket under 100"}
        )

        self.assertEqual(
            self.persistence.load_preferences(conversation)["budget"], 100
        )
        self.assertEqual(
            self.persistence.load_search_history(conversation)[0]["query"],
            "lightweight rain jacket under 100",
        )
        self.assertEqual(
            self.persistence.load_agent_state(conversation)["last_query"],
            "lightweight rain jacket under 100",
        )

    def test_memory_context_is_compact(self):
        conversation = self.persistence.create_conversation()
        self.persistence.save_message(conversation, "user", "I like black")
        context = self.persistence.build_memory_context(conversation)
        self.assertEqual(context["recent_messages"][0]["content"], "I like black")
        self.assertIn("preferences", context)
        self.assertIn("recent_searches", context)
        self.assertIn("agent_state", context)

    def test_similarity_analysis_stores_query_and_product_vectors(self):
        conversation = self.persistence.create_conversation()
        search_id = self.persistence.save_search(
            conversation, "waterproof hiking jacket", ["Shopify"], []
        )
        self.persistence.save_similarity_analysis(
            conversation,
            search_id,
            "waterproof hiking jacket",
            [1.0, 0.0],
            [
                {
                    "item_key": "shopify:jacket",
                    "text": "title: Rain jacket",
                    "embedding": [0.8, 0.6],
                    "score": 0.8,
                    "rank": 1,
                    "product": {"title": "Rain jacket"},
                }
            ],
            "gemini-embedding-001",
        )

        with self.persistence.connection() as conn:
            embeddings = conn.execute(
                """
                SELECT item_type, task_type, embedding_json
                FROM search_embeddings
                WHERE search_id = ?
                ORDER BY id
                """,
                (search_id,),
            ).fetchall()
            scores = conn.execute(
                """
                SELECT cosine_similarity, rank
                FROM similarity_scores
                WHERE search_id = ?
                """,
                (search_id,),
            ).fetchall()

        self.assertEqual(
            [(row["item_type"], row["task_type"]) for row in embeddings],
            [("query", "RETRIEVAL_QUERY"), ("product", "RETRIEVAL_DOCUMENT")],
        )
        self.assertEqual(json.loads(embeddings[0]["embedding_json"]), [1.0, 0.0])
        self.assertEqual(json.loads(embeddings[1]["embedding_json"]), [0.8, 0.6])
        self.assertEqual(
            [(row["cosine_similarity"], row["rank"]) for row in scores],
            [(0.8, 1)],
        )

    def test_export_contains_only_the_requested_conversation(self):
        requested = self.persistence.create_conversation()
        private = self.persistence.create_conversation()
        self.persistence.save_message(requested, "user", "Find a rain jacket")
        self.persistence.save_message(private, "user", "Find running shoes")
        for conversation, query, title in (
            (requested, "Find a rain jacket", "Rain jacket"),
            (private, "Find running shoes", "Running shoes"),
        ):
            search_id = self.persistence.save_search(conversation, query, ["Shopify"], [])
            self.persistence.save_similarity_analysis(
                conversation,
                search_id,
                query,
                [1.0, 0.0],
                [
                    {
                        "item_key": title,
                        "text": f"title: {title}",
                        "embedding": [0.8, 0.6],
                        "score": 0.8,
                        "rank": 1,
                        "product": {"title": title},
                    }
                ],
                "gemini-embedding-001",
            )

        export_path = Path(self.tmp.name) / "export.db"
        export_path.write_bytes(self.persistence.export_conversation_db(requested))
        conn = sqlite3.connect(export_path)
        try:
            conversations = conn.execute(
                "SELECT id FROM conversations"
            ).fetchall()
            messages = conn.execute(
                "SELECT content FROM messages ORDER BY id"
            ).fetchall()
            embeddings = conn.execute(
                "SELECT item_type, text_content FROM search_embeddings ORDER BY id"
            ).fetchall()
            scores = conn.execute(
                "SELECT cosine_similarity, rank FROM similarity_scores"
            ).fetchall()
        finally:
            conn.close()

        self.assertEqual(conversations, [(requested,)])
        self.assertEqual(messages, [("Find a rain jacket",)])
        self.assertEqual(
            embeddings,
            [("query", "Find a rain jacket"), ("product", "title: Rain jacket")],
        )
        self.assertEqual(scores, [(0.8, 1)])


if __name__ == "__main__":
    unittest.main()
