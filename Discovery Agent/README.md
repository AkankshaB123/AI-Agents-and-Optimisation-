# Shopify Discovery Assistant — System Design Architecture

## Executive Summary
The **Shopify Discovery Assistant** is a multi-tenant product discovery engine. It uses an LLM (Llama 3.3 70B via Hugging Face Router) for natural language intent parsing and dynamic tool invocation, combining parallel Shopify Store JSON searches with DuckDuckGo fallback web browsing.

---

## 1. System Architecture Overview

The system is built as a single-node interactive assistant using Streamlit for UI and session orchestration. It coordinates intent parsing, session state persistence, multi-threaded e-commerce querying, and dynamic HTML carousel rendering.

```
+----------------+      +--------------------------+      +------------------------------+
|                |      | Intent & Routing Engine  |----->| Shopify Parallel Workers     |---> [Shopify Stores]
|  Streamlit UI  |----->| - Regex Rule Parser      |      | (ThreadPoolExecutor)         |
|  (User Chat)   |      | - Llama 3.3 70B Agent    |      +------------------------------+
+----------------+      +--------------------------+                     |
                                     |                    +------------------------------+
                                     +------------------->| DDG Web Search Workers       |---> [Web Search]
                                                          | (DuckDuckGo Search API)      |
                                                          +------------------------------+
```

---

## 2. Core Component Breakdown

* **Rule-Based Parsing Engine (*Deterministic*)**: Uses regular expressions to extract structured parameters (price caps, gender requirements, product category synonyms, target brand detection) prior to LLM reasoning.
* **LLM Agent & Tool Calling (*Dynamic*)**: Employs Hugging Face Router with `Llama-3.3-70B-Instruct`. Enforces category rules, detects user ambiguity, and decides when to call `search_shopify_dynamic`.
* **Parallel Store Search Engine (*I/O Bound*)**: Leverages Python's `ThreadPoolExecutor` with up to 25 workers. Direct queries hit Shopify endpoints (`/search/suggest.json`) with low timeouts (1.2s).
* **Web Search Fallback (*Adaptive*)**: Utilizes `duckduckgo_search` for explicitly named external brands (e.g., Nike, Adidas) or as a fallback to ensure broad product discovery coverage.

---

## 3. Request Lifecycle & Intent Routing Flow

```
[ User Input ]
      │
      ▼
[ 1. Regex Rule Parser ] ─── (Extracts budget, gender, categories, brand flags)
      │
      ▼
[ 2. Llama 3.3 70B LLM Agent ] ─── (Evaluates context & decides tool execution)
      │
      ├─── Call `search_shopify_dynamic()`
      │          │
      │          ├─── Direct External Brand Match? ───> [ DuckDuckGo Worker ]
      │          │
      │          └─── General Discovery Query ───────> [ ThreadPoolExecutor (25 Threads) ]
      │                                                         │
      │                                                         ▼
      │                                              [ Query 20+ Shopify Stores ]
      │
      ▼
[ 3. Deduplication & Filtering Engine ] ─── (Max 1 product/store, apply price/gender caps)
      │
      ▼
[ 4. Dynamic HTML Carousel Render ]
```

---

## 4. Module Specifications & Key Methods

| Function / Component | Input / Dependencies | Primary Responsibility & Logic |
| :--- | :--- | :--- |
| `parse_user_prompt` | Raw `user_prompt` string | Executes regex mapping for prices (e.g., `under $50`), genders, category synonyms, and brand detection against explicit and store lists. |
| `fetch_store_products` | Store domain, query, filter params | Calls `https://{store}/search/suggest.json`. Enforces in-stock checking, synonym/category matching, gender filtering, and budget caps. |
| `fetch_duckduckgo_products` | Search query string, brand name | Queries DDG text and image endpoints concurrently. Parses price markers from snippets and pairs them with image thumbnails. |
| `search_shopify_dynamic` | Cleaned search prompt string | Main router. Redirects external brands to DDG or dispatches parallel workers across 20+ Shopify stores with DDG backup. Deduplicates by store. |
| `display_carousel_html` | Normalized products list | Transforms product JSON objects into styled horizontal scrolling cards with images, stock badges, pricing, and direct links. |

---

## 5. Intent & Filtering Constraints Matrix

* **Category Synonyms Filtering**: Enforces strict item type matches. E.g., `t-shirt` matches keywords: *t-shirt, tshirt, tee, tees*. Prevents irrelevant store search suggestions from leaking into user results.
* **Gender Isolation Rules**: If `female` is active, titles containing *men's, mens, for men* are explicitly dropped. If `male` is active, *women's, womens, for women* are excluded.

---

## 6. Key Operational & Performance Metrics

* **HTTP API Timeout**: `1.2s` per store endpoint
* **Max Parallel Threads**: `25` concurrent workers (`ThreadPoolExecutor`)
* **Max Displayed Cards**: `6` unique store product items
* **Context Window Truncation**: Conversation limited to the last `6` chat turns (`conversation[-6:]`)

---

## 7. System Risk & Recommended Enhancements

| Area | Current Bottleneck / Risk | Proposed Architectural Enhancement |
| :--- | :--- | :--- |
| **Rate Limiting** | High-frequency Shopify endpoint queries may lead to IP blocking or HTTP 429 errors. | Implement an asynchronous proxy rotation system or store caching layer (e.g., Redis with a 15-min TTL). |
| **DDG Dependency** | Web scraping DDG snippets for prices is prone to formatting inconsistencies. | Integrate structured commercial APIs (e.g., Google Shopping API or SerpAPI) for unified product parsing. |
| **State Management** | Session state is ephemeral and isolated to a single server process instance. | Migrate session storage and user preferences to an external persistence store (e.g., PostgreSQL / Redis). |
| **Model Latency** | Synchronous LLM completion requests create visible UI loading spinners. | Implement streaming responses for chat text and asynchronously hydrate the product carousel component. |

---

## 8. Supported Stores Overview
Pre-configured storefront endpoints:
`gymshark.com`, `allbirds.com`, `kith.com`, `fabletics.com`, `aloyoga.com`, `chubbieshorts.com`, `rothys.com`, `taylormade-golf.com`, `cotopaxi.com`, `vuoriclothing.com`, `rhone.com`, `marine-layer.com`, `mottandbow.com`, `bombas.com`, `noble-apparel.com`, `unTuckit.com`, `brooklinen.com`, `outerknown.com`, `tentree.com`, `blundstone.com`, `tuckernuck.com`.