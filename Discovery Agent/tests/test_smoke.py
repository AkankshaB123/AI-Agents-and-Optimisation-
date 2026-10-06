import sys
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))


class DiscoveryAgentSmokeTests(unittest.TestCase):
    def test_imports_and_search_tool(self):
        from src.product_search import build_search_tool, get_enabled_providers

        providers = get_enabled_providers()
        self.assertIn("Shopify", providers)
        tool = build_search_tool(providers)
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "search_shopify_dynamic")

    def test_required_entrypoint_exists(self):
        self.assertTrue((APP_DIR / "streamlit_app.py").is_file())


if __name__ == "__main__":
    unittest.main()
