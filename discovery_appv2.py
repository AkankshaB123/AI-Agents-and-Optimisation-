"""Compatibility entrypoint for existing Streamlit Cloud settings."""

import sys
from pathlib import Path


APP_DIRECTORY = Path(__file__).resolve().parent / "Discovery Agent"
sys.path.insert(0, str(APP_DIRECTORY))

from streamlit_app import run_app


run_app()