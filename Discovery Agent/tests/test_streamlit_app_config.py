import streamlit_app


def test_configured_gemini_api_key_reads_streamlit_secret(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        streamlit_app.st, "secrets", {"GEMINI_API_KEY": "secret-api-key"}
    )

    assert streamlit_app._configured_gemini_api_key() == "secret-api-key"


def test_environment_gemini_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", " environment-api-key ")
    monkeypatch.setattr(
        streamlit_app.st, "secrets", {"GEMINI_API_KEY": "secret-api-key"}
    )

    assert streamlit_app._configured_gemini_api_key() == "environment-api-key"
