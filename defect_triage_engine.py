"""
Defect Triage & Semantic Similarity Engine
Bridges ChromaDB vector embeddings and Gemini LLM to provide:
1. Semantic search over historical Jira defects.
2. Automated Root Cause Analysis (RCA) and triage recommendations.
"""

import os
import re
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress unauthenticated HF Hub warning and noisy library loggers
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
for _log_name in ("httpx", "httpx2", "httpcore", "httpcore2", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_log_name).setLevel(logging.WARNING)

logger = logging.getLogger("defect_triage_engine")

# Base directory for Chroma DB
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHROMA_DIR = BASE_DIR / "Defect Triage" / "chroma_db"
if not DEFAULT_CHROMA_DIR.exists():
    DEFAULT_CHROMA_DIR = BASE_DIR / "chroma_db"

DEFAULT_COLLECTION = "jira_defects"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

_embedding_model = None
_chroma_client = None


def get_embedding_model():
    """Lazily loads and caches the SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            hf_token = os.getenv("HF_TOKEN") or False
            _embedding_model = SentenceTransformer(model_name, token=hf_token)
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer: {e}")
            raise
    return _embedding_model


def get_chroma_collection(db_dir: Optional[Path] = None, collection_name: str = DEFAULT_COLLECTION):
    """Retrieves the persistent Chroma collection for defects."""
    global _chroma_client
    target_dir = Path(db_dir) if db_dir else DEFAULT_CHROMA_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"ChromaDB directory not found at: {target_dir}")

    try:
        import chromadb
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=str(target_dir))
        return _chroma_client.get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Error accessing ChromaDB collection '{collection_name}': {e}")
        raise


def extract_document_title(document: str) -> str:
    """Extracts title line from embedded text."""
    if document.startswith("Title:"):
        title = document.split("\n\nDescription:", 1)[0]
        return title.removeprefix("Title:").strip()
    return ""


def search_similar_defects(
    query_text: str,
    top_k: int = 3,
    db_dir: Optional[Path] = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search across historical Jira defects stored in ChromaDB.
    Returns matched defects with similarity scores, document contents, and known root causes.
    """
    query_text = (query_text or "").strip()
    if not query_text:
        return []

    collection = get_chroma_collection(db_dir=db_dir, collection_name=collection_name)
    total_count = collection.count()
    if total_count == 0:
        raise RuntimeError(f"Chroma collection '{collection_name}' is empty. Run defect embedding first.")

    model = get_embedding_model()
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    result_count = min(top_k, total_count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for record_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = max(0.0, min(1.0, 1.0 - float(dist)))
            matches.append({
                "defect_id": record_id,
                "title": extract_document_title(doc),
                "document": doc,
                "root_cause": (meta or {}).get("root_cause", "") if meta else "",
                "similarity": round(sim, 4),
            })
    return matches


def build_triage_prompt(issue_key: str, summary: str, description: str, matches: List[Dict[str, Any]]) -> str:
    """Constructs prompt with historical evidence for Gemini defect triage."""
    historical_context = []
    for idx, m in enumerate(matches, start=1):
        defect_id = m.get("defect_id", f"UNKNOWN-{idx}")
        similarity = m.get("similarity", 0.0) * 100
        details = m.get("document") or m.get("description") or "No details provided"
        root_cause = m.get("root_cause") or "Not documented"
        historical_context.append(
            f"Similar Defect #{idx} (Key: {defect_id}, Similarity: {similarity:.1f}%):\n"
            f"Details: {details}\n"
            f"Known Root Cause: {root_cause}"
        )

    evidence_text = "\n\n".join(historical_context) if historical_context else "No historical matches found."

    prompt = (
        "You are an expert software defect triage assistant and site reliability engineer.\n\n"
        "Analyze the following current Jira defect using the retrieved historical defect evidence.\n"
        "Do not invent facts that are not supported by the available evidence.\n\n"
        f"--- CURRENT JIRA DEFECT ---\n"
        f"Key: {issue_key}\n"
        f"Title: {summary}\n"
        f"Description: {description or 'No description provided'}\n\n"
        f"--- HISTORICAL SIMILAR DEFECT EVIDENCE ---\n"
        f"{evidence_text}\n\n"
        "Provide a clear, actionable triage recommendation formatted in markdown with these exact sections:\n"
        "### 1. Likely Root Cause\n"
        "### 2. Evidence from Similar Historical Defects\n"
        "### 3. Recommended Investigation & Fix Steps\n"
        "### 4. Confidence Level (Low / Medium / High) and Justification"
    )
    return prompt


def run_defect_triage(
    issue_key: str,
    summary: str,
    description: str,
    matches: List[Dict[str, Any]],
    gemini_api_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
) -> str:
    """Invokes Gemini LLM to generate the triage recommendation with multi-model cascade failover."""
    key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        # Fallback if no Gemini key is provided
        top_cause = matches[0].get("root_cause", "Unknown") if matches else "Unknown"
        top_match_desc = (
            f"{matches[0].get('defect_id', 'Unknown')} ({matches[0].get('similarity', 0.0)*100:.1f}% match)"
            if matches else "None"
        )
        return (
            f"### Triage Summary (Heuristic Fallback - No GEMINI_API_KEY)\n"
            f"- **Issue Key**: {issue_key}\n"
            f"- **Top Similar Defect**: {top_match_desc}\n"
            f"- **Known Root Cause**: {top_cause}\n\n"
            f"*(Add GEMINI_API_KEY in .env to enable full generative AI root cause analysis)*"
        )

    # Build model cascade list
    models_to_try = []
    if gemini_model:
        models_to_try.append(gemini_model.strip())
    else:
        env_models = os.getenv("GEMINI_MODELS", "")
        if env_models:
            models_to_try.extend([m.strip() for m in env_models.split(",") if m.strip()])
        single_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        if single_model and single_model not in models_to_try:
            models_to_try.insert(0, single_model)
    if not models_to_try:
        models_to_try = [DEFAULT_GEMINI_MODEL, "gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash"]

    try:
        from google import genai
        client = genai.Client(api_key=key)
        prompt = build_triage_prompt(issue_key, summary, description, matches)

        last_error = None
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text
            except Exception as me:
                err_str = str(me).lower()
                logger.warning(f"Gemini model '{model_name}' failed during triage: {me}. Checking cascade...")
                last_error = me
                if any(x in err_str for x in ["503", "unavailable", "overloaded", "spikes in demand", "429", "quota", "resource_exhausted", "500", "502", "504"]):
                    continue
                # If non-retryable error (e.g., bad auth), abort loop
                break

        logger.error(f"All Gemini models in cascade failed: {last_error}")
        return f"Error generating triage recommendation via Gemini (attempted: {', '.join(models_to_try)}): {last_error}"

    except Exception as e:
        logger.error(f"Gemini client setup or execution failed: {e}")
        return f"Error generating triage recommendation via Gemini: {e}"
