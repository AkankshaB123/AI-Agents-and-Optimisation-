"""Streamlit UI entrypoint for the Shopify Discovery Assistant."""

import json
import os
import uuid
from html import escape

import streamlit as st
from openai import OpenAI

from src.product_search import (
    build_search_tool,
    get_enabled_providers,
    search_shopify_dynamic,
)


st.set_page_config(
    page_title="Shopify Discovery Assistant", page_icon="🛍️", layout="wide"
)

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct:together"
EXIT_INTENTS = [
    "exit",
    "quit",
    "bye",
    "goodbye",
    "good bye",
    "no need now",
    "no need",
    "stop",
    "close",
]

SYSTEM_PROMPT = """You are an expert personal shopping assistant routing natural-language product searches to connected catalogs.

Search naturally:
- Accept conversational product requests and preserve the shopper's stated preferences, budget, use case, style, gender, and brand in the search prompt.
- When a clear maximum budget is provided, pass it as `max_price` so the catalog applies a hard price filter in the configured currency.
- Do NOT require the shopper to choose a category, brand, marketplace, or structured filter.
- Select `providers` based on explicit marketplace intent and query meaning. If the shopper explicitly says eBay or Amazon, prefer only that provider. Use eBay for used/auction/listing intent. Use Amazon for Amazon-specific shopping/Associate product intent. Use Shopify for broad cross-merchant Shopify discovery. Use Google only for searches expected to match the configured retailer-owned Google catalog, never as a public Google Shopping search.
- If the shopper explicitly requests a marketplace that is not in the available-provider list, explain that it is not configured and ask before searching a different provider.
- For general requests without a marketplace preference, search relevant connected providers; avoid selecting providers unrelated to the request.
- Available providers for this session: {enabled_providers}.
- Never claim a provider is connected unless it appears in the available-provider list; never imply these catalogs include every merchant on the internet.
- Ask a follow-up only when the request is too ambiguous to identify a useful search, not merely because a category or filter is missing.
- Do not invent product facts, prices, stock, or merchant details; rely on catalog results.

Tool rules:
- Invoke `search_shopify_dynamic` for requests to find or compare products.
- Pass a concise but faithful natural-language search phrase that retains useful constraints. Extract an explicit maximum budget into `max_price`; do not invent a budget. Include one or more provider names from the available-provider list.
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


def reset_session(enabled_providers: list):
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                enabled_providers=", ".join(enabled_providers)
            ),
        },
        {
            "role": "assistant",
            "content": "Hi! What can I help you find today?",
            "products": [],
        },
    ]
    st.session_state.session_closed = False


def run_app():
    st.title("🛍️ Personal Shopping Assistant")
    st.write("Describe what you want in your own words. I’ll search connected shopping catalogs.")
    enabled_providers = get_enabled_providers()
    st.caption(f"Connected catalogs: {', '.join(enabled_providers)}")

    if "session_closed" not in st.session_state:
        st.session_state.session_closed = False
    if "catalog_visitor_id" not in st.session_state:
        st.session_state.catalog_visitor_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        reset_session(enabled_providers)
    else:
        st.session_state.messages[0]["content"] = SYSTEM_PROMPT.format(
            enabled_providers=", ".join(enabled_providers)
        )

    hf_token = st.sidebar.text_input(
        "Enter Hugging Face Token:",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        key="hf_token_input",
    )
    if not hf_token:
        st.warning(
            "🔑 Please enter your Hugging Face Token in the sidebar to start the chat session."
        )
        return

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

    with st.sidebar:
        st.header("Session Status")
        if st.session_state.session_closed:
            st.warning("Session Closed")
            if st.button("🔄 Start New Session"):
                reset_session(enabled_providers)
                st.rerun()
        else:
            st.success("Bot Active")
            if st.button("❌ End Session"):
                st.session_state.session_closed = True
                st.rerun()

    if st.session_state.session_closed:
        st.info("👋 **Session Ended.** Your chat session is closed.")
        st.write(
            "To start a new session, re-open/refresh the Streamlit link or click **'Start New Session'** in the sidebar."
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
        return

    cleaned_input = user_input.strip().lower()
    if any(intent in cleaned_input for intent in EXIT_INTENTS):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Happy shopping! Session closed. Re-open the page or click 'Start New Session' in the sidebar anytime.",
                "products": [],
            }
        )
        st.session_state.session_closed = True
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    conversation = [
        message for message in st.session_state.messages[1:] if message.get("content")
    ]
    payload_messages = [st.session_state.messages[0]] + [
        {"role": message["role"], "content": message["content"]}
        for message in conversation[-6:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Choosing catalogs and searching..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=payload_messages,
                    tools=[build_search_tool(enabled_providers)],
                    tool_choice="auto",
                )
                assistant_message = response.choices[0].message
                products = []

                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        if tool_call.function.name == "search_shopify_dynamic":
                            arguments = json.loads(tool_call.function.arguments)
                            products = search_shopify_dynamic(
                                **arguments,
                                visitor_id=st.session_state.catalog_visitor_id,
                            )

                    if products:
                        reply_text = f"Here are top matching results for **'{user_input}'**:"
                    else:
                        reply_text = "I couldn't find matching products in the connected catalogs. Try another description or broaden your request."

                    st.markdown(reply_text)
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
                else:
                    reply_text = assistant_message.content
                    st.markdown(reply_text)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply_text,
                            "products": [],
                        }
                    )
            except Exception as error:
                st.error(f"Error executing search request: {error}")


if __name__ == "__main__":
    run_app()
