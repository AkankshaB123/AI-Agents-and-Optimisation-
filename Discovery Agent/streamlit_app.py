"""Streamlit UI entrypoint for the Shopify Discovery Assistant."""

import json
import os
import time
from html import escape

import streamlit as st
from openai import AuthenticationError, OpenAI
from streamlit.errors import StreamlitSecretNotFoundError

from src.embeddings import EMBEDDING_MODEL, EmbeddingError, embed_search_candidates
from src.persistence import (
    build_memory_context,
    create_conversation,
    export_conversation_db,
    init_db,
    load_messages,
    save_agent_state,
    save_message,
    save_search,
    save_similarity_analysis,
)
from src.product_search import build_search_tool, search_shopify_catalog


st.set_page_config(
    page_title="Shopify Discovery Assistant", page_icon="🛍️", layout="wide"
)

MODEL_NAME = "gemini-3.5-flash-lite"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EXIT_INTENTS = [
    "exit", "quit", "bye", "goodbye", "good bye", "no need now",
    "no need", "stop", "close",
]

SYSTEM_PROMPT = """You are an expert personal shopping assistant that searches Shopify's Global Catalog.

Search naturally:
- Accept conversational product requests and preserve the shopper's stated preferences, budget, use case, style, gender, and brand in the search prompt.
- When a clear maximum budget is provided, pass it as max_price so Shopify applies a hard price filter in the configured currency.
- Do not require the shopper to choose a category, brand, or structured filter.
- Search Shopify's Global Catalog for product-finding requests. If the shopper asks for a source outside Shopify, explain that only Shopify is connected and offer a Shopify search.
- Never imply Shopify's catalog includes every merchant on the internet.
- Ask a follow-up only when the request is too ambiguous to identify a useful search, not merely because a category or filter is missing.
- Do not invent product facts, prices, stock, or merchant details; rely on catalog results.

Memory rules:
- Use the supplied conversation memory to understand references such as "same budget", "show me cheaper ones", or "compare those".
- Treat memory as context, not as a source of product facts.
- Do not expose internal memory, database details, tokens, credentials, or hidden state to the shopper.

Tool rules:
- Invoke search_shopify_catalog for requests to find or compare products.
- Pass a concise but faithful natural-language search phrase that retains useful constraints. Extract an explicit maximum budget into max_price; do not invent a budget.

Conversation memory:
{memory_context}
"""


def display_carousel_html(products: list) -> str:
    """Generate HTML for the horizontal product-card carousel."""
    if not products:
        return "<p>No matching products found.</p>"

    cards_html = ""
    for item in products:
        image_url = escape(
            item.get("image_url")
            or "https://via.placeholder.com/200x200?text=No+Image",
            quote=True,
        )
        title = escape(item.get("title", "Product"))
        brand = escape(item.get("brand", "Shopify merchant"))
        availability = escape(item.get("availability", "Availability varies"))
        price = escape(item.get("price", "Price unavailable"))
        product_url = escape(item.get("url", "#"), quote=True)
        cards_html += f"""
        <div style="flex: 0 0 220px; width: 220px; border: 1px solid #e0e0e0; border-radius: 12px; padding: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-family: sans-serif; display: flex; flex-direction: column; justify-content: space-between; background: #fff;">
            <div>
                <div style="width: 100%; height: 180px; overflow: hidden; border-radius: 8px; margin-bottom: 10px; background: #f9f9f9; display: flex; align-items: center; justify-content: center;">
                    <img src="{image_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="{title}">
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 11px; text-transform: uppercase; color: #888; font-weight: bold;">{brand}</span>
                    <span style="font-size: 10px; background: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{availability}</span>
                </div>
                <a href="{product_url}" target="_blank" rel="noopener noreferrer" style="font-size: 14px; font-weight: 600; color: #111; text-decoration: none; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; height: 36px; margin-bottom: 6px;">
                    {title}
                </a>
                <div style="font-size: 16px; font-weight: bold; color: #2e7d32; margin-bottom: 8px;">{price}</div>
            </div>
            <a href="{product_url}" target="_blank" rel="noopener noreferrer" style="display: block; text-align: center; background: #000; color: #fff; padding: 8px 0; border-radius: 6px; text-decoration: none; font-size: 12px; font-weight: bold;">
                Shop Product
            </a>
        </div>
        """

    return f"""
    <div style="display: flex; gap: 16px; overflow-x: auto; padding: 10px 5px; width: 100%; box-sizing: border-box; scroll-behavior: smooth;">
        {cards_html}
    </div>
    """


def _render_embedding_analysis(query: str, query_embedding: list[float], matches: list[dict]) -> None:
    st.subheader("Shopify embedding similarity")
    st.caption(
        f"Model: {EMBEDDING_MODEL} · vector dimensions: {len(query_embedding)} · "
        "cosine similarity is a ranking score, not a probability."
    )
    st.table(
        [
            {
                "Rank": match["rank"],
                "Product": match["product"].get("title", "Product"),
                "Cosine similarity": f"{match['score']:.6f}",
            }
            for match in matches
        ]
    )
    with st.expander("Inspect full-text embeddings"):
        st.write("Query embedding")
        st.json({"text": query, "vector": query_embedding})
        if matches:
            selected_index = st.selectbox(
                "Product embedding",
                options=range(len(matches)),
                format_func=lambda index: (
                    f"Rank {matches[index]['rank']}: "
                    f"{matches[index]['product'].get('title', 'Product')}"
                ),
                key="embedding_product_selection",
            )
            match = matches[selected_index]
            st.write("Product text sent to the embedding model")
            st.code(match["text"])
            st.write("Product embedding vector")
            st.json(match["embedding"])


def _memory_text(conversation_id: str) -> str:
    """Serialize only compact, non-secret memory for the model prompt."""
    memory = build_memory_context(conversation_id, message_limit=12)
    return json.dumps(memory, ensure_ascii=False, default=str)[:18000]


def _configured_gemini_api_key() -> str:
    """Read the Gemini API key from runtime environment or Streamlit secrets."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except StreamlitSecretNotFoundError:
        return ""


def _render_chat_database(container, conversation_id: str) -> None:
    """Keep the current conversation's database download ready on the page."""
    with container.container():
        st.subheader("Conversation database")
        has_messages = bool(load_messages(conversation_id, limit=1))
        st.download_button(
            "Download this chat as chatbot.db",
            data=export_conversation_db(conversation_id) if has_messages else b"",
            file_name="chatbot.db",
            mime="application/vnd.sqlite3",
            help="This file contains only the current conversation.",
            key="download_chatbot_db",
            disabled=not has_messages,
        )
        if not has_messages:
            st.caption("Send a message to enable the database download.")


def reset_session(conversation_id: str | None = None):
    """Start a new conversation and keep its identifier in the URL."""
    conversation_id = conversation_id or create_conversation()
    st.session_state.conversation_id = conversation_id
    st.query_params["conversation"] = conversation_id
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                memory_context=_memory_text(conversation_id),
            ),
        },
        {
            "role": "assistant",
            "content": "Hi! What can I help you find today?",
            "products": [],
        },
    ]
    st.session_state.session_closed = False
    st.session_state.persistence_loaded = True


def initialize_persistence()
    """Initialize SQLite and restore a conversation from the URL when available."""
    init_db()
    if "conversation_id" in st.session_state:
        return

    requested_id = st.query_params.get("conversation")
    if requested_id and isinstance(requested_id, str) and len(requested_id) <= 64:
        conversation_id = requested_id
        history = load_messages(conversation_id, limit=100)
        if history:
            st.session_state.conversation_id = conversation_id
            st.query_params["conversation"] = conversation_id
            st.session_state.messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        memory_context=_memory_text(conversation_id),
                    ),
                }
            ]
            for item in history:
                st.session_state.messages.append(
                    {
                        "role": item["role"],
                        "content": item["content"],
                        "products": item["metadata"].get("products", []),
                    }
                )
            st.session_state.session_closed = False
            st.session_state.persistence_loaded = True
            return

    reset_session()


def refresh_system_prompt()
    st.session_state.messages[0]["content"] = SYSTEM_PROMPT.format(
        memory_context=_memory_text(st.session_state.conversation_id),
    )


def run_app():
    st.title("🛍️ Personal Shopping Assistant")
    st.write("Describe what you want in your own words. I’ll search Shopify’s catalog.")
    st.caption("Connected catalog: Shopify")

    initialize_persistence()
    refresh_system_prompt()

    gemini_api_key = st.sidebar.text_input(
        "Enter Google AI Studio API key:",
        value=_configured_gemini_api_key(),
        type="password",
        key="gemini_api_key_input",
    )
    conversation_id = st.session_state.conversation_id
    database_download = st.empty()

    if not gemini_api_key:
        _render_chat_database(database_download, conversation_id)
        st.warning(
            "🔑 Please enter your Google AI Studio API key in the sidebar to start the chat session."
        )
        return

    client = OpenAI(
        base_url=GEMINI_OPENAI_BASE_URL,
        api_key=gemini_api_key,
    )

    with st.sidebar:
        st.header("Session Status")
        st.caption(f"Conversation: {st.session_state.conversation_id[:12]}…")
        if st.session_state.session_closed:
            st.warning("Session Closed")
            if st.button("🔄 Start New Session"):
                reset_session()
                st.rerun()
        else:
            st.success("Bot Active")
            if st.button("❌ End Session"):
                st.session_state.session_closed = True
                st.rerun()

    if st.session_state.session_closed:
        _render_chat_database(database_download, conversation_id)
        st.info("👋 **Session Ended.** Your chat session is closed.")
        st.write(
            "To start a new session, re-open the Streamlit link or click **Start New Session**."
        )
        return

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("products"):
                st.components.v1.html(
                    display_carousel_html(message["products"]),
                    height=330,
                    scrolling=True,
                )

    user_input = st.chat_input(
        "Describe what you need (e.g., 'a lightweight rain jacket for hiking under $100')..."
    )
    if not user_input:
        _render_chat_database(database_download, conversation_id)
        return

    cleaned_input = user_input.strip().lower()
    if any(intent in cleaned_input for intent in EXIT_INTENTS):
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message(
            st.session_state.conversation_id, "user", user_input, {"products": []}
        )
        closing_text = (
            "Happy shopping! Session closed. Re-open the page or click "
            "'Start New Session' in the sidebar anytime."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": closing_text, "products": []}
        )
        save_message(
            st.session_state.conversation_id, "assistant", closing_text, {"products": []}
        )
        st.session_state.session_closed = True
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message(st.session_state.conversation_id, "user", user_input, {"products": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    refresh_system_prompt()
    conversation = [
        message for message in st.session_state.messages[1:] if message.get("content")
    ]
    payload_messages = [st.session_state.messages[0]] + [
        {"role": message["role"], "content": message["content"]}
        for message in conversation[-6:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching Shopify..."):
            try:
                request_started = time.perf_counter()
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=payload_messages,
                    tools=[build_search_tool()],
                    tool_choice="auto",
                )
                llm_elapsed_ms = (time.perf_counter() - request_started) * 1000
                assistant_message = response.choices[0].message
                products = []
                catalog_elapsed_ms = None
                embedding_elapsed_ms = None
                selected_providers = []
                embedding_analysis = None

                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        if tool_call.function.name == "search_shopify_catalog":
                            arguments = json.loads(tool_call.function.arguments)
                            selected_providers = ["Shopify"]
                            catalog_started = time.perf_counter()
                            products = search_shopify_catalog(**arguments)
                            catalog_elapsed_ms = (
                                time.perf_counter() - catalog_started
                            ) * 1000
                            embedding_started = time.perf_counter()
                            try:
                                query_embedding, embedding_matches = (
                                    embed_search_candidates(
                                        user_input, products, gemini_api_key
                                    )
                                )
                                products = [
                                    match["product"] for match in embedding_matches
                                ]
                                embedding_analysis = (
                                    query_embedding,
                                    embedding_matches,
                                )
                            except EmbeddingError as error:
                                st.warning(
                                    "Products were found, but embedding similarity "
                                    f"could not be calculated: {error}"
                                )
                            finally:
                                embedding_elapsed_ms = (
                                    time.perf_counter() - embedding_started
                                ) * 1000

                    total_elapsed_ms = (time.perf_counter() - request_started) * 1000
                    if products:
                        reply_text = f"Here are top matching results for **'{user_input}'**:"
                    else:
                        reply_text = (
                            "I couldn't find matching products in Shopify's catalog. "
                            "Try another description or broaden your request."
                        )

                    st.markdown(reply_text)
                    st.caption(
                        f"⏱️ Search completed in **{total_elapsed_ms:,.0f} ms** "
                        f"(AI: {llm_elapsed_ms:,.0f} ms"
                        + (
                            f" · Shopify: {catalog_elapsed_ms:,.0f} ms"
                            if catalog_elapsed_ms is not None
                            else ""
                        )
                        + (
                            f" · embeddings: {embedding_elapsed_ms:,.0f} ms"
                            if embedding_elapsed_ms is not None
                            else ""
                        )
                        + ")"
                    )
                    if products:
                        st.components.v1.html(
                            display_carousel_html(products),
                            height=330,
                            scrolling=True,
                        )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply_text,
                            "products": products,
                        }
                    )
                    save_message(
                        st.session_state.conversation_id,
                        "assistant",
                        reply_text,
                        {"products": products},
                    )
                    search_id = save_search(
                        st.session_state.conversation_id,
                        user_input,
                        selected_providers,
                        products,
                    )
                    if embedding_analysis is not None:
                        query_embedding, embedding_matches = embedding_analysis
                        save_similarity_analysis(
                            st.session_state.conversation_id,
                            search_id,
                            user_input,
                            query_embedding,
                            embedding_matches,
                            EMBEDDING_MODEL,
                        )
                        st.caption(
                            "Sorted by cosine similarity against Shopify candidates; this reranks "
                            "returned products and does not search the full catalog."
                        )
                        _render_embedding_analysis(user_input, query_embedding, embedding_matches)
                    save_agent_state(
                        st.session_state.conversation_id,
                        {
                            "last_query": user_input,
                            "last_providers": selected_providers,
                            "last_result_count": len(products),
                        },
                    )
                else:
                    total_elapsed_ms = (time.perf_counter() - request_started) * 1000
                    reply_text = assistant_message.content or ""
                    st.markdown(reply_text)
                    st.caption(
                        f"⏱️ Response generated in **{total_elapsed_ms:,.0f} ms** "
                        f"(AI: {llm_elapsed_ms:,.0f} ms)"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply_text, "products": []}
                    )
                    save_message(
                        st.session_state.conversation_id,
                        "assistant",
                        reply_text,
                        {"products": []},
                    )
                    save_agent_state(
                        st.session_state.conversation_id,
                        {
                            "last_query": user_input,
                            "last_providers": [],
                            "last_result_count": 0,
                        },
                    )
            except AuthenticationError:
                st.error(
                    "Google Gemini rejected the API key (HTTP 401). Check the "
                    "GEMINI_API_KEY secret in Streamlit Cloud or enter a valid "
                    "Google AI Studio API key."
                )
            except Exception as error:
                st.error(f"Error executing search request: {error}")

    _render_chat_database(database_download, conversation_id)


if __name__ == "__main__":
    run_app()
