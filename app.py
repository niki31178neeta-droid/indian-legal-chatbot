from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer

# Keep the three Excel files in the same folder as this app.py file.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = {
    "IP": BASE_DIR / "MILPaC_IP_dataset.xlsx",
    "CCI_FAQ": BASE_DIR / "MILPaC_CCI_FAQ_dataset.xlsx",
    "Acts": BASE_DIR / "MILPaC_Acts_dataset.xlsx",
}
BASE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FINETUNED_MODEL_DIR = BASE_DIR / "milpac_legal_embedding_finetuned"
MODEL_NAME = str(FINETUNED_MODEL_DIR) if (FINETUNED_MODEL_DIR / "modules.json").exists() else BASE_MODEL_NAME
CACHE_FILE = BASE_DIR / ".milpac_semantic_index.pkl"

LANGUAGES = {
    "EN": "English",
    "HI": "Hindi",
    "BN": "Bengali",
    "GU": "Gujarati",
    "ML": "Malayalam",
    "MR": "Marathi",
    "OR": "Odia",
    "PA": "Punjabi",
    "TA": "Tamil",
    "TE": "Telugu",
}



def read_table(path: Path) -> pd.DataFrame:
    """Read either the supplied Excel files or equivalent CSV files."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path.name}. Put all three MILPaC files in this folder: {BASE_DIR}"
        )
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    else:
        frame = pd.read_excel(path, dtype=str)
    frame = frame.fillna("")
    required = {"id", "src_lang", "src", "tgt_lang", "tgt"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing_columns))}")
    return frame


def source_signature() -> str:
    details: dict[str, Any] = {"model": MODEL_NAME}
    model_config = FINETUNED_MODEL_DIR / "modules.json"
    if model_config.exists():
        details["fine_tuned_model_mtime_ns"] = model_config.stat().st_mtime_ns
    for dataset, path in DATA_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path.name}. Put all three MILPaC files next to app.py."
            )
        stat = path.stat()
        details[dataset] = {"file": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return json.dumps(details, sort_keys=True)


def build_items() -> dict[str, dict[str, dict[str, Any]]]:
    """Group rows by dataset and item ID, retaining English and available translations."""
    items: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, path in DATA_FILES.items():
        frame = read_table(path)
        grouped: dict[str, dict[str, Any]] = {}
        for row in frame.itertuples(index=False):
            item_id = str(getattr(row, "id", "")).strip()
            if not item_id:
                continue
            record = grouped.setdefault(
                item_id,
                {"dataset": dataset, "id": item_id, "english": "", "translations": {}},
            )
            english_text = str(getattr(row, "src", "")).strip()
            if english_text and not record["english"]:
                record["english"] = english_text
            language = str(getattr(row, "tgt_lang", "")).strip().upper()
            translation = str(getattr(row, "tgt", "")).strip()
            if language and translation:
                record["translations"][language] = translation
        for record in grouped.values():
            if record["english"]:
                record["translations"]["EN"] = record["english"]
        items[dataset] = grouped
    return items


def text_for(record: dict[str, Any], language: str) -> tuple[str, str]:
    """Return (text, actual language code), falling back to English if needed."""
    translations = record.get("translations", {})
    if translations.get(language):
        return translations[language], language
    if record.get("english"):
        return record["english"], "EN"
    return "", language


def create_search_documents(items: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    # IP and CCI FAQ entries use matching IDs such as Q1/A1 and Q_1_1/A_1_1.
    for dataset in ("IP", "CCI_FAQ"):
        dataset_items = items.get(dataset, {})
        for question_id, question_record in dataset_items.items():
            if not question_id.upper().startswith("Q"):
                continue
            answer_id = "A" + question_id[1:]
            if answer_id not in dataset_items:
                continue
            answer_record = dataset_items[answer_id]
            for language, question_text in question_record["translations"].items():
                question_text = str(question_text).strip()
                if not question_text:
                    continue
                docs.append(
                    {
                        "kind": "faq",
                        "key": f"{dataset}::{question_id}",
                        "dataset": dataset,
                        "item_id": question_id,
                        "answer_id": answer_id,
                        "language": language,
                        "search_text": question_text,
                    }
                )

    # Acts are indexed as searchable legal provisions, using each available translation.
    for act_id, act_record in items.get("Acts", {}).items():
        for language, provision_text in act_record["translations"].items():
            provision_text = str(provision_text).strip()
            if not provision_text:
                continue
            docs.append(
                {
                    "kind": "act",
                    "key": f"Acts::{act_id}",
                    "dataset": "Acts",
                    "item_id": act_id,
                    "language": language,
                    "search_text": provision_text,
                }
            )
    return docs


@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """Load the multilingual model and build/cache embeddings for the local datasets."""
    signature = source_signature()
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("signature") == signature:
                return model, cached["items"], cached["docs"], cached["embeddings"]
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError, KeyError):
            # A damaged or old cache is rebuilt below.
            pass

    items = build_items()
    docs = create_search_documents(items)
    if not docs:
        raise ValueError("No searchable text was found. Check the dataset columns and contents.")

    texts = [doc["search_text"] for doc in docs]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    try:
        with CACHE_FILE.open("wb") as handle:
            pickle.dump(
                {"signature": signature, "items": items, "docs": docs, "embeddings": embeddings},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except OSError:
        # The app still works if the folder is read-only; the index then stays in RAM.
        pass
    return model, items, docs, embeddings


def retrieve(query: str, model, items, docs, embeddings, limit: int = 8):
    query_vector = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0].astype(np.float32)
    scores = embeddings @ query_vector

    # Several translations belong to the same question or Act item. Keep only its best score.
    best_by_key: dict[str, dict[str, Any]] = {}
    for index, score in enumerate(scores):
        doc = docs[index]
        candidate = {**doc, "score": float(score)}
        prior = best_by_key.get(doc["key"])
        if prior is None or candidate["score"] > prior["score"]:
            best_by_key[doc["key"]] = candidate

    ranked = sorted(best_by_key.values(), key=lambda entry: entry["score"], reverse=True)
    faq_hits = [hit for hit in ranked if hit["kind"] == "faq"]
    act_hits = [hit for hit in ranked if hit["kind"] == "act"]
    return faq_hits[:limit], act_hits[:limit]


def build_reply(query: str, output_language: str, threshold: float, model, items, docs, embeddings):
    faq_hits, act_hits = retrieve(query, model, items, docs, embeddings)
    parts: list[str] = []

    best_faq = faq_hits[0] if faq_hits else None
    if best_faq and best_faq["score"] >= threshold:
        dataset = best_faq["dataset"]
        question_record = items[dataset][best_faq["item_id"]]
        answer_record = items[dataset][best_faq["answer_id"]]
        question_text, question_language = text_for(question_record, output_language)
        answer_text, answer_language = text_for(answer_record, output_language)
        parts.append("### Closest matching FAQ")
        if question_text:
            parts.append(f"**Question:** {question_text}")
        parts.append(answer_text or "No answer text was found for this FAQ entry.")
        note = f"Source: `{dataset}` · question ID `{best_faq['item_id']}` · semantic score `{best_faq['score']:.3f}`"
        if answer_language != output_language:
            note += f"\n\n*The answer is not available in {LANGUAGES.get(output_language, output_language)} in this dataset, so English is shown.*"
        parts.append(f"*{note}*")
    else:
        parts.append(
            "I could not find a FAQ match above the selected similarity threshold. "
            "I have listed the closest searchable Act provisions below, if any."
        )

    relevant_acts = [hit for hit in act_hits if hit["score"] >= max(0.22, threshold - 0.07)][:3]
    if relevant_acts:
        parts.append("### Related Act text from your dataset")
        for hit in relevant_acts:
            act_record = items["Acts"][hit["item_id"]]
            provision, actual_language = text_for(act_record, output_language)
            if not provision:
                continue
            parts.append(
                f"**Acts item `{hit['item_id']}` · score `{hit['score']:.3f}`**\n\n{provision}"
            )
            if actual_language != output_language:
                parts.append(
                    f"*This item has no stored {LANGUAGES.get(output_language, output_language)} translation; English is shown.*"
                )

    if not relevant_acts and (not best_faq or best_faq["score"] < threshold):
        parts.append("Try rephrasing the question, using a legal keyword, or changing the answer language.")

    parts.append(
        "---\n**Important:** This prototype retrieves text from the supplied MILPaC datasets; it does not verify current law or replace advice from a qualified lawyer. Similarity scores are ranking signals, not legal-confidence probabilities."
    )
    return "\n\n".join(parts)


def make_evidence(query, output_language, threshold, model, items, docs, embeddings):
    faq_hits, act_hits = retrieve(query, model, items, docs, embeddings, limit=5)
    blocks = []
    sources = []
    for hit in faq_hits[:3]:
        if hit["score"] < threshold:
            continue
        dataset = hit["dataset"]
        qrec = items[dataset].get(hit["item_id"], {})
        arec = items[dataset].get(hit["answer_id"], {})
        qtext, qlang = text_for(qrec, output_language)
        atext, alang = text_for(arec, output_language)
        if not atext:
            atext, alang = text_for(arec, "EN")
        if atext:
            label = f"{dataset} FAQ {hit['item_id']} (retrieval score {hit['score']:.3f})"
            blocks.append(f"SOURCE: {label}\nQUESTION: {qtext or qrec.get('english','')}\nANSWER: {atext}")
            sources.append({"label": label, "question": qtext, "answer": atext, "score": hit["score"]})
    for hit in act_hits[:4]:
        if hit["score"] < max(0.22, threshold - 0.07):
            continue
        rec = items["Acts"].get(hit["item_id"], {})
        passage, actual_lang = text_for(rec, output_language)
        if not passage:
            passage, actual_lang = text_for(rec, "EN")
        if passage:
            label = f"MILPaC Acts item {hit['item_id']} (retrieval score {hit['score']:.3f})"
            blocks.append(f"SOURCE: {label}\nTEXT: {passage}")
            sources.append({"label": label, "question": "", "answer": passage, "score": hit["score"]})
    return blocks, sources


def generate_with_ollama(prompt: str, model_name: str = "gemma3:4b") -> str:
    """Call Gemma running locally through Ollama."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.2, "num_ctx": 4096}},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_with_hf(prompt: str, token: str, model_name: str = "google/gemma-3-4b-it") -> str:
    """Call the hosted Hugging Face OpenAI-compatible Inference Providers endpoint."""
    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model_name,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2, "max_tokens": 700},
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"Hugging Face inference returned {response.status_code}: {response.text[:500]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def make_prompt(question: str, language: str, evidence: list[str]) -> str:
    language_name = LANGUAGES.get(language, "English")
    evidence_text = "\n\n---\n\n".join(evidence)
    return f"""You are a careful multilingual legal information assistant for an academic prototype.
Answer in {language_name}. The user may ask in any language supported by the application.
Use ONLY the evidence provided below for claims about law. Do not invent statutes, section numbers,
procedures, deadlines, penalties, or legal remedies. If the evidence is insufficient, say so clearly
in {language_name}. Explain the material in plain language, distinguish source text from explanation,
and do not claim to be a lawyer. Keep the answer concise. Mention the source labels that support the answer.

USER QUESTION:
{question}

RETRIEVED MILPaC EVIDENCE:
{evidence_text if evidence_text else 'No relevant source was retrieved. State that the dataset does not contain enough evidence to answer.'}

ANSWER:"""


def main():
    st.set_page_config(page_title="MILPaC Multilingual Legal Assistant", page_icon="⚖️", layout="wide")
    st.title("⚖️ MILPaC Multilingual Legal Assistant")
    st.caption("Multilingual semantic retrieval + source-grounded answer generation using your MILPaC datasets.")
    with st.sidebar:
        st.header("Settings")
        output_language = st.selectbox("Answer language", options=list(LANGUAGES), index=0,
                                       format_func=lambda code: f"{LANGUAGES[code]} ({code})")
        threshold = st.slider("Minimum FAQ match score", 0.15, 0.65, 0.32, 0.01,
                              help="Heuristic retrieval threshold, not a probability of correctness.")
        runtime = st.radio("Answer generation", ["Hosted Gemma (website)", "Local Gemma via Ollama", "Retrieval only"], index=0)
        st.markdown("**Knowledge sources**")
        st.write("MILPaC IP FAQ · CCI FAQ · Acts")
        st.caption("Do not enter confidential personal/legal case details in this academic demo.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    try:
        with st.spinner("Loading multilingual MiniLM and indexing MILPaC documents…"):
            model, items, docs, embeddings = load_knowledge_base()
    except Exception as exc:
        st.error(f"Could not load the knowledge base: {exc}")
        st.info("Keep all three MILPaC Excel files beside app.py and check requirements.txt.")
        st.stop()
    st.success(f"Knowledge base ready: {len(docs):,} searchable translated text entries.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a legal question in English, Hindi, Gujarati, or another supported language…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant FAQs and legal passages…"):
                evidence, sources = make_evidence(question, output_language, threshold, model, items, docs, embeddings)
            prompt = make_prompt(question, output_language, evidence)
            answer = ""
            generation_error = None
            if runtime == "Hosted Gemma (website)":
                try:
                    token = st.secrets.get("HF_TOKEN", "")
                except Exception:
                    token = ""
                if token:
                    try:
                        with st.spinner("Gemma is generating an answer from the retrieved evidence…"):
                            answer = generate_with_hf(prompt, token)
                    except Exception as exc:
                        generation_error = str(exc)
                else:
                    generation_error = "No HF_TOKEN is configured. Add it in Streamlit Community Cloud → App settings → Secrets."
            elif runtime == "Local Gemma via Ollama":
                try:
                    with st.spinner("Local Gemma is generating an answer…"):
                        answer = generate_with_ollama(prompt)
                except Exception as exc:
                    generation_error = f"Could not reach local Ollama/Gemma: {exc}"
            else:
                generation_error = "Retrieval-only mode selected."

            if answer:
                st.markdown(answer)
                if sources:
                    with st.expander("Retrieved evidence and source IDs", expanded=True):
                        for source in sources:
                            st.markdown(f"**{source['label']}**")
                            if source["question"]:
                                st.markdown(f"Question: {source['question']}")
                            st.write(source["answer"])
                            st.divider()
            else:
                if generation_error:
                    st.warning(f"Generative answer unavailable: {generation_error}")
                st.markdown("### Retrieved information from MILPaC")
                if sources:
                    for source in sources:
                        st.markdown(f"**{source['label']}**")
                        if source["question"]:
                            st.markdown(f"Question: {source['question']}")
                        st.write(source["answer"])
                        st.divider()
                else:
                    st.info("No sufficiently relevant passage was found. Try rephrasing the question or use a legal keyword.")
                answer = "Retrieved source material is shown above. A generated explanation was not available in this run."
            st.caption("Academic prototype only. Similarity scores are not confidence probabilities. Verify legal information against current official sources; this is not legal advice.")
        st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("Architecture used"):
        st.markdown("""
        1. **Input and preprocessing:** the user's question is tokenized by the multilingual embedding model.
        2. **MiniLM encoder:** maps the query into a 384-dimensional embedding.
        3. **Semantic retrieval:** compares it with precomputed embeddings for translated MILPaC FAQ and Act passages.
        4. **Evidence assembly:** retrieves the best matching FAQ answers and Act text, retaining dataset item IDs.
        5. **Gemma 3 4B Instruct:** in hosted or local mode, generates a plain-language answer using the question and retrieved evidence.
        6. **Answer with sources:** the interface shows retrieved evidence for verification. If generation is unavailable, the app falls back to source text.

        The website mode calls a hosted inference API; local mode calls Ollama running on your own computer. The hosted mode needs a Hugging Face access token and model/provider availability. The retrieval-only mode needs no generative API.
        """)


if __name__ == "__main__":
    main()
