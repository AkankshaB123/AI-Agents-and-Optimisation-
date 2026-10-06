import os
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


if __name__ == "__main__":
    unittest.main()
