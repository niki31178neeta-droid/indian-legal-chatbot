# MILPaC Multilingual Legal Assistant — Jury Demo

This Streamlit application follows the supplied flow: multilingual user question → MiniLM encoder → 384-dimensional embedding → semantic search of MILPaC → retrieved FAQs/Act passages → Gemma 3 4B Instruct generation → answer with source references.

## Files
- `app.py`: Streamlit app, semantic retrieval and answer generation
- `MILPaC_IP_dataset.xlsx`, `MILPaC_CCI_FAQ_dataset.xlsx`, `MILPaC_Acts_dataset.xlsx`: your datasets
- `requirements.txt`: dependencies
- `.streamlit/secrets.toml.example`: template only; do not put a real token in GitHub

## A. Run on your laptop (Windows)
1. Install Python 3.10 or 3.11 and install [Ollama](https://ollama.com/).
2. Open a terminal in this folder and run:
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull gemma3:4b
streamlit run app.py
```
3. Choose **Local Gemma via Ollama** in the sidebar. The first run downloads the embedding model and indexes the datasets. Gemma 3 4B is a quantized local model; performance depends on your RAM/CPU. If it is too slow or memory is insufficient, use **Retrieval only**.

## B. Publish a live website for your jury (recommended: Streamlit Community Cloud)
1. Create a GitHub repository, e.g. `milpac-legal-assistant`.
2. Upload `app.py`, `requirements.txt`, and all three `.xlsx` datasets to the repository root. Do not upload `.venv`, cache files, model weights or real secrets.
3. Open https://share.streamlit.io/ and connect GitHub. Choose **Create app**, select your repository, branch `main`, and entrypoint `app.py`, then deploy. Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy
4. For generated answers on the hosted website, create a Hugging Face access token after accepting the Gemma model terms if required. In the deployed app, open **App settings → Secrets** and add:
```toml
HF_TOKEN = "hf_your_real_token"
```
5. Restart/redeploy. Select **Hosted Gemma (website)**. The hosted mode calls Hugging Face's OpenAI-compatible Inference Providers endpoint for `google/gemma-3-4b-it`. It requires that your token has access and that the model is available through a provider for your account. Provider availability/quotas can change. Test it before the jury.
6. Share the resulting `https://....streamlit.app` URL with your guide. Streamlit assigns a live URL after successful deployment.

If hosted inference is unavailable, choose **Retrieval only**; the website will still retrieve and show matching dataset answers/passages. You can alternatively use a different hosted LLM provider by adapting the `generate_with_hf` function.

## How to evaluate
Prepare a held-out set of manually verified questions in each target language. Report Recall@k/MRR for retrieval, answer correctness and source faithfulness for generation, language quality, and response latency. Do not treat similarity score as probability.

## Legal safety
This is an academic prototype, not a substitute for legal advice. Retrieved passages may be incomplete, translated text may contain errors, and the app does not independently verify current law. Confirm answers against current official sources. Avoid entering confidential case details.
