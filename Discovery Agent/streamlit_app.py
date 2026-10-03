"""Streamlit UI entrypoint for the Shopify Discovery Assistant."""

import json
import os

import streamlit as st
from openai import OpenAI

from src.product_search import search_shopify_dynamic


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

SYSTEM_PROMPT = """You are an expert personal shopping assistant.

Category & Intent Mandate Rules:
- You MUST analyze and adhere strictly to category-level intent. If the user asks for a specific category (e.g., "t-shirt"), ONLY return products belonging strictly to that category.
- Persist active user constraints (gender, explicit brand, category) across the session.
- If an explicit brand is requested by the user (e.g., "Nike"), ensure that brand is explicitly searched.

Ambiguity & Confidence Rules:
- If the user's request is vague, ambiguous, broad, or lacks concrete product detail (e.g., "show me stuff", "something cool"), DO NOT invoke any tools. Respond directly asking: "I want to help you find the right item! Could you be a bit more specific about what you are looking for (e.g., Nike gymwear, women's t-shirt)?"

Tool Triggering Rules:
- Invoke `search_shopify_dynamic` ONLY when the user expresses clear intent for a concrete product category or type.
- Clean the `prompt` parameter: strip out conversational filler and supply core query keywords along with any explicit category/gender/brand modifiers.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_shopify_dynamic",
            "description": "Searches store endpoints and DuckDuckGo web search with smart intent routing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Cleaned search query keywords including active gender, brand, and category parameters.",
                    }
                },
                "required": ["prompt"],
            },
        },
    }
]


def display_carousel_html(products: list) -> str:
    """Generate HTML for the horizontal product-card carousel."""
    if not products:
        return "<p>No matching products found.</p>"

    cards_html = ""
    for item in products:
        image_url = (
            item.get("image_url")
            or "https://via.placeholder.com/200x200?text=No+Image"
        )
        cards_html += f"""
        <div style="flex: 0 0 220px; width: 220px; border: 1px solid #e0e0e0; border-radius: 12px; padding: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-family: sans-serif; display: flex; flex-direction: column; justify-content: space-between; background: #fff;">
            <div>
                <div style="width: 100%; height: 180px; overflow: hidden; border-radius: 8px; margin-bottom: 10px; background: #f9f9f9; display: flex; align-items: center; justify-content: center;">
                    <img src="{image_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="{item['title']}">
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 11px; text-transform: uppercase; color: #888; font-weight: bold;">{item['brand']}</span>
                    <span style="font-size: 10px; background: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{item['availability']}</span>
                </div>
                <a href="{item['url']}" target="_blank" style="font-size: 14px; font-weight: 600; color: #111; text-decoration: none; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; height: 36px; margin-bottom: 6px;">
                    {item['title']}
                </a>
                <div style="font-size: 16px; font-weight: bold; color: #2e7d32; margin-bottom: 8px;">{item['price']}</div>
            </div>
            <a href="{item['url']}" target="_blank" style="display: block; text-align: center; background: #000; color: #fff; padding: 8px 0; border-radius: 6px; text-decoration: none; font-size: 12px; font-weight: bold;">
                View Product
            </a>
        </div>
        """

    return f"""
    <div style="display: flex; gap: 16px; overflow-x: auto; padding: 10px 5px; width: 100%; box-sizing: border-box; scroll-behavior: smooth;">
        {cards_html}
    </div>
    """


def reset_session():
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": "Hi! What can I help you find today?",
            "products": [],
        },
    ]
    st.session_state.user_intent = {
        "gender": None,
        "brand": None,
        "category": None,
    }
    st.session_state.session_closed = False


def run_app():
    st.title("🛍️ Personal Shopping Assistant")
    st.write("Search across 20+ top Shopify brands and web stores simultaneously!")

    if "session_closed" not in st.session_state:
        st.session_state.session_closed = False
    if "messages" not in st.session_state:
        reset_session()

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
        if st.session_state.get("user_intent"):
            st.subheader("Active Preferences")
            st.write(
                f"**Category Mandate:** {st.session_state.user_intent.get('category') or 'None'}"
            )
            st.write(
                f"**Gender:** {st.session_state.user_intent.get('gender') or 'Not specified'}"
            )
            st.write(
                f"**Brand Lock:** {st.session_state.user_intent.get('brand') or 'All Stores'}"
            )

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
        "Ask for items (e.g., 'nike brand gymwear', 'women's t-shirts under $50')..."
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
        with st.spinner("Searching stores and web..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=payload_messages,
                    tools=TOOLS,
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
                                user_intent=st.session_state.user_intent,
                            )

                    if products:
                        reply_text = f"Here are top matching results for **'{user_input}'**:"
                    else:
                        reply_text = "No results found matching your specific category, availability, and intent criteria."

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
