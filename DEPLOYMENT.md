# Discovery Agent CI/CD deployment

## Architecture

GitHub is the source of truth. Every pull request runs the Discovery Agent CI workflow. Merges to main run the same validation and are automatically picked up by Streamlit Community Cloud for deployment.

Streamlit Community Cloud coordinates the runtime deployment; GitHub Actions is responsible for repeatable validation before changes reach main.

## Streamlit Community Cloud settings

Use these exact coordinates when creating or recreating the app:

- Repository: AkankshaB123/AI-Agents-and-Optimisation-
- Branch: main
- Main file: Discovery Agent/streamlit_app.py
- Python: 3.11

The repository now has a root requirements.txt and root .streamlit/config.toml, which removes ambiguity caused by the app living in a subdirectory.

## Secrets

Do not commit credentials. In Streamlit Community Cloud, add these through the app's Secrets settings:

HF_TOKEN = "<your Hugging Face token>"

Optional provider credentials from Discovery Agent/.env.example can be added as needed. Shopify is enabled by default and does not require a credential in this code path.

## Release flow

1. Create a feature branch.
2. Open a pull request into main.
3. GitHub Actions compiles the app and runs smoke tests.
4. Merge only after CI is green.
5. Streamlit Community Cloud automatically detects the main branch update and redeploys the app.
6. Verify the app and inspect Streamlit Cloud logs if the runtime fails.

## Local validation

From the repository root, install requirements, compile the four Python modules, run unittest discovery under Discovery Agent/tests, then run Streamlit with Discovery Agent/streamlit_app.py as the entrypoint.
