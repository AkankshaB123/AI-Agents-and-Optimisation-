# Shopify Discovery Assistant

A Streamlit shopping assistant that uses Gemini for conversational product discovery and Shopify's Global Catalog MCP as its single catalog source. Whole-text Gemini embeddings rerank returned Shopify candidates by cosine similarity, and vectors/scores are persisted in SQLite.

## Live app

[Open the Shopify Discovery Assistant](https://ksjamznbsx6brz9rtobt8i.streamlit.app/). Set the app secret `GEMINI_API_KEY` or enter it in the sidebar. A successful search displays a rank/product/cosine-similarity table and an expander with the full query vector and an inspected product's text and vector. Scores are ranking values, not probabilities. They rerank results returned by Shopify; the app does not download or index the entire catalog.

## Run locally

Use Python 3.11 or newer from this directory.

1. Create a virtual environment and install dependencies:

   `python3.11 -m venv .venv`
   `source .venv/bin/activate`
   `python -m pip install -r requirements.txt`

2. Configure `GEMINI_API_KEY` in your environment or Streamlit secrets. Copy `.env.example` for supported settings. Shopify market country/currency default to `US`/`USD`.
3. Run `streamlit run streamlit_app.py`.

Never commit API keys. The Gemini key is used for chat and embeddings; Shopify's Global Catalog MCP integration requires no Shopify API credential.

## Chat database and embeddings

The app initializes `data/chatbot.db` and persists conversations, messages, search history, embeddings, and similarity scores. Once you have sent a message, the main page enables **Download this chat as chatbot.db**, which exports only the current conversation. Open it in a SQLite viewer. Configure `DISCOVERY_DB_PATH` to use another local database path.

Vectors are whole-text embeddings, not individual token embeddings. Query/product vectors are stored as JSON in `search_embeddings.embedding_json`; candidate scores and ranks are in `similarity_scores` and link to `search_history`. To inspect matches:

```sql
SELECT h.query, e.text_content, s.cosine_similarity, s.rank
FROM similarity_scores AS s
JOIN search_history AS h ON h.id = s.search_id
JOIN search_embeddings AS e ON e.id = s.product_embedding_id
ORDER BY h.id DESC, s.rank;
```

Cloud app storage is separate from your local database and may be ephemeral; durable shared history requires an external persistent database.

## Tests

From this directory, install pytest if needed and run:

```bash
python -m compileall -q src streamlit_app.py tests
python -m pytest -q
```

## CI/CD

Pull requests run the repository's GitHub Actions validation and tests. A merge to `main` validates and publishes the container to GHCR; Streamlit Community Cloud is configured to redeploy from `main`. See [DEPLOYMENT.md](../DEPLOYMENT.md) for deployment settings and release steps.
