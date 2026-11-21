# Discovery Agent CI/CD deployment

GitHub is the source of truth. Pull requests run validation and tests. The repository's GitHub Actions workflow publishes the Discovery Agent container to GitHub Container Registry (GHCR) on pushes to `main` and version tags. Streamlit Community Cloud is configured to redeploy from `main` when it changes.

## Streamlit Community Cloud

Configure the app with:

- Repository: `AkankshaB123/AI-Agents-and-Optimisation-`
- Branch: `main`
- Main file: `Discovery Agent/streamlit_app.py`
- Python: 3.11

Add `GEMINI_API_KEY` to the app's Secrets settings. This key is required for Gemini chat and embeddings. Shopify Global Catalog MCP does not require credentials. Optional market settings are `SHOPIFY_CATALOG_COUNTRY` and `SHOPIFY_CATALOG_CURRENCY`; the defaults are `US` and `USD`. `SHOPIFY_UCP_AGENT_PROFILE` can override the default hosted agent profile. Do not put credentials in the repository or container image.

## Release flow

1. Create a feature branch and open a pull request into `main`.
2. Wait for the GitHub Actions validation and test workflows to pass; resolve any failures.
3. Merge the reviewed pull request.
4. On the push to `main`, GitHub Actions reruns validation and publishes the container image. Streamlit Community Cloud detects the update to `main` and redeploys.
5. Confirm the app starts and that a search displays product results and embedding diagnostics. Inspect Streamlit Cloud logs if deployment fails.

## Container deployment

The repository workflow publishes images to `ghcr.io/akankshab123/discovery-agent`. Use the commit SHA or version tag for a pinned release; `latest` tracks `main`. Configure `GEMINI_API_KEY` in your container platform's secret manager.

For local checks, from `Discovery Agent/` install `requirements.txt`, then run `python -m compileall -q src streamlit_app.py tests` and `python -m pytest -q`. See the [Discovery Agent README](./Discovery%20Agent/README.md) for local setup and database/embedding inspection.
