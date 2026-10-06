"""Streamlit Cloud entrypoint for the Discovery Agent."""
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent / "Discovery Agent"
sys.path.insert(0, str(APP_DIR))

from streamlit_app import run_app


if __name__ == "__main__":
    run_app()
