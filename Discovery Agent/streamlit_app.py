"""Streamlit UI entrypoint for the Shopify Discovery Assistant."""


import json
import os
import time
import uuid
from html import escape


import streamlit as st
from openai import OpenAI


from src.persistence import (
    build_memory_context,
    create_conversation,
    init_db,
    load_messages,
    save_agent_state,
    save_message,
    save_search,
)
from src.product_search import (
    build_search_tool,
    get_enabled_providers,
    search_shopify_dynamic,
)




st.set_page_config(
    page_title="Shopify Discovery Assistant", page_icon="🛍️", layout="wide"
)


MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct:novita"
EXIT_INTENTS = [
    "exit", "quit", "bye", "goodbye", "good bye", "no need now",
    "no need", "stop", "close",
]
