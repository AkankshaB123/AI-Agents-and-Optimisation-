# Shopify Discovery Assistant — System Design Architecture

## Executive Summary
The **Shopify Discovery Assistant** uses an LLM (Llama 3.3 70B via Hugging Face Router) to understand conversational product requests and Shopify's Global Catalog MCP (UCP) to search eligible Shopify merchants in one catalog call.

## Live dashboard

Open the deployed [Shopify Discovery Assistant dashboard](https://ksjamznbsx6brz9rtobt8i.streamlit.app/). The app may take a short time to wake up if it has been idle. Enter a Hugging Face access token in the sidebar to start chatting. Describe what you want conversationally; categories and structured filters are optional.

Browse the project files on [GitHub](https://github.com/AkankshaB123/AI-Agents-and-Optimisation-/tree/main/Discovery%20Agent).

## Run locally

The Streamlit UI is `streamlit_app.py`; reusable search and intent code lives under `src/`. Use Python 3.11 or newer.

1. Create and activate a virtual environment from the repository root:

      ```bash
      python3.11 -m venv .venv
      source .venv/bin/activate
      ```

2. Install the app dependencies:

      ```bash
      python -m pip install --upgrade pip
      python -m pip install -r requirements.txt
      ```

3. Launch the app:

      ```bash
      streamlit run streamlit_app.py
      ```

4. Open the local URL printed by Streamlit (usually `http://localhost:8501`) and enter your Hugging Face token in the sidebar. Keep the token private; do not commit it to the repository. You can also set `HF_TOKEN` in your environment before launching.

### Run checks

Install `pytest` if it is not already available, then run the unit tests and Python compilation checks:

```bash
python -m pip install pytest
python -m compileall -q src streamlit_app.py tests
python -m pytest -q
```

The tests cover MCP request construction and catalog result normalization. Live product searches also require a valid Hugging Face token and network access to the Hugging Face Router and Shopify Global Catalog MCP.

---

## 1. System Architecture Overview

The system uses Streamlit for UI and session orchestration, a hosted LLM for conversational intent, and Shopify Global Catalog MCP for cross-merchant product discovery.

```
[ Streamlit UI ] -> [ Llama 3.3 intent + optional budget extraction ]
                           |
                           v
                [ Shopify Global Catalog MCP / UCP ]
                           |
                           v
                [ Normalize product offers ] -> [ Product cards ]
```

---

## 2. Core Component Breakdown

* **Natural-language intent parsing**: The model retains the shopper's free-text constraints (budget, activity, style, gender, or brand) without making category selection mandatory.
* **LLM Agent & Tool Calling (*Dynamic*)**: Employs Hugging Face Router with `Llama-3.3-70B-Instruct` and calls the catalog search tool with the shopper's natural-language request.
* **Global Catalog MCP**: Sends a single UCP-compatible MCP `search_catalog` request to Shopify's Global Catalog endpoint. Results span eligible Shopify merchants and include seller, variant, availability, price, media, and product links.

---

## 3. Request Lifecycle & Intent Routing Flow

```
[ Shopper's natural-language request ]
      -> [ Llama 3.3: retains intent and extracts an explicit max budget ]
      -> [ Shopify Global Catalog MCP / UCP: search_catalog ]
      -> [ Normalize seller offers, currency, availability, and product links ]
      -> [ Render product cards ]
```

---

## 4. Module Specifications & Key Methods

| Function / Component | Input / Dependencies | Primary Responsibility & Logic |
| :--- | :--- | :--- |
| `build_catalog_request` | Natural-language prompt and optional market context | Builds a UCP-compatible MCP `tools/call` request to the global `search_catalog` tool. |
| `normalize_catalog_response` | MCP Global Catalog response | Converts available catalog variants, seller data, currency minor units, media, and safe HTTPS product links into UI cards. |
| `search_shopify_dynamic` | Natural-language search prompt | Calls Shopify Global Catalog MCP once and returns normalized result cards. |
| `display_carousel_html` | Normalized products list | Transforms product JSON objects into styled horizontal scrolling cards with images, stock badges, pricing, and direct links. |

---

## 5. Intent & Filtering Constraints Matrix

* **Natural-language intent**: Shoppers can describe product type, use case, brand, style, and other preferences in free text; selecting structured categories is not required.
* **Hard price filtering**: If the shopper states a maximum budget, the assistant extracts it into the catalog's `filters.price.max` using the configured currency's minor units.
* **Availability filtering**: Catalog searches default to available products; result cards are additionally normalized using variant availability.

---

## 6. Key Operational & Performance Metrics

* **Catalog requests**: One Global Catalog MCP search request per product lookup
* **Default market context**: United States / USD, configurable with environment variables
* **Max Displayed Cards**: `6` product offers
* **Context Window Truncation**: Conversation limited to the last `6` chat turns (`conversation[-6:]`)

---

## 7. System Risk & Recommended Enhancements

| Area | Current Bottleneck / Risk | Proposed Architectural Enhancement |
| :--- | :--- | :--- |
| **Catalog coverage** | Global Catalog covers eligible Shopify listings, not every online merchant. | Add a separately licensed search provider only if broader non-Shopify coverage is a requirement. |
| **Catalog latency and limits** | A single catalog API avoids fanning out to many stores, but latency and rate limits remain service-dependent. | Measure end-to-end latency and handle timeouts/rate-limit responses; don't cache product search results or images. |
| **State Management** | Session state is ephemeral and isolated to a single server process instance. | Migrate session storage and user preferences to an external persistence store (e.g., PostgreSQL / Redis). |
| **Model Latency** | Synchronous LLM completion requests create visible UI loading spinners. | Implement streaming responses for chat text and asynchronously hydrate the product carousel component. |

---

## 8. Supported Stores Overview
The app now searches Shopify's eligible Global Catalog rather than a hard-coded store list. Catalog availability and results depend on Shopify's eligibility, market context, and catalog response.

## MLOps CI/CD

The repository-root [GitHub Actions workflow](../.github/workflows/discovery-agent.yml) validates Python syntax, lint, and unit tests for pull requests and pushes. On pushes to `main` and version tags (`v*`), it builds the Streamlit container and publishes commit-, release-, and (on `main`) `latest`-tagged images to GitHub Container Registry (GHCR).

The Llama model is served remotely through the Hugging Face Router; this repository does not train or package model weights. Product discovery uses Shopify's Global Catalog MCP endpoint (`https://catalog.shopify.com/api/ucp/mcp`) and a UCP agent profile. The pipeline versions and delivers the application container. Configure `HF_TOKEN` as a runtime secret in the container platform. The app also supports entering the token in its sidebar for local interactive use. Set `SHOPIFY_CATALOG_COUNTRY` and `SHOPIFY_CATALOG_CURRENCY` to change the default market (`US`/`USD`); `SHOPIFY_UCP_AGENT_PROFILE` can override the example profile with an agent profile you host.

Application code is separated into `src/` (`discovery_logic.py` and `product_search.py`); `streamlit_app.py` is the independent Streamlit UI entrypoint. The previous app scripts are preserved under `notebooks/legacy/`. See [Run locally](#run-locally) for setup instructions.

Compatibility launchers named `discovery_appv2.py` remain at the repository root and in this folder for existing Streamlit Cloud settings; both delegate to the maintained `streamlit_app.py` UI.

### Run locally with Docker

Build the image with `docker build -t discovery-assistant .`, then start it with `docker run --rm -p 8501:8501 -e HF_TOKEN="$HF_TOKEN" discovery-assistant`. Open `http://localhost:8501`.

For deployment, pull the published image `ghcr.io/<owner>/<repository>:<tag>` into your container platform and inject `HF_TOKEN` through that platform's secret manager. Do not put tokens in the image or commit them to the repository.

## Other product catalog options

Shopify Global Catalog MCP is a good match for cross-merchant discovery specifically across eligible Shopify products. Other providers serve different catalog scopes and usually require separate credentials, program approval, or merchant-owned feeds:

* [eBay Browse API](https://developer.ebay.com/api-docs/buy/browse/overview.html): search eBay listings by keyword, category, product identifier, or image; requires an eBay application access token.
* [Amazon Creators API](https://affiliate-program.amazon.com/creatorsapi/docs/en-us/introduction): Amazon product catalog access for eligible publishers, influencers, and affiliate partners. Amazon says this is the successor to the deprecated PA-API 5.
* [Google Merchant API](https://developers.google.com/merchant/api): manage and read products in your own Google Merchant Center account; it is not a general cross-store shopping-search API.
* Hosted search providers such as Algolia or Constructor: useful for a retailer's own product index, but they do not automatically provide a global multi-merchant catalog.

For this app, keep Shopify Global Catalog as the primary Shopify discovery source. Consider adding eBay or an approved Amazon integration only if you want those marketplaces represented too; normalize each provider into the same product-card format and respect its access and display terms.