"""Streamlit application for Jira defect triage using ChromaDB and Gemini."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_DIR = BASE_DIR / "chroma_db" if (BASE_DIR / "chroma_db").exists() else Path("./chroma_db")
DEFAULT_COLLECTION = "jira_defects"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
TOP_K = 3


def get_config() -> dict[str, str]:
	load_dotenv(BASE_DIR / ".env")
	load_dotenv()
	config = {
		"jira_base_url": os.getenv("JIRA_BASE_URL", "").strip().rstrip("/"),
		"jira_email": os.getenv("JIRA_EMAIL", "").strip(),
		"jira_api_token": os.getenv("JIRA_API_TOKEN", "").strip(),
		"gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
		"gemini_model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip(),
	}
	missing = [
		name for name in ("jira_base_url", "jira_email", "jira_api_token", "gemini_api_key")
		if not config[name]
	]
	if missing:
		raise RuntimeError(
			"Missing configuration in .env: " + ", ".join(missing)
		)
	return config


def jira_value_to_text(value: Any) -> str:
	"""Convert Jira plain text or Atlassian Document Format to readable text."""
	if value is None:
		return ""
	if isinstance(value, str):
		return value.strip()
	if isinstance(value, list):
		return "\n".join(
			part for part in (jira_value_to_text(item) for item in value) if part
		).strip()
	if not isinstance(value, dict):
		return str(value).strip()
	if value.get("type") == "text":
		return str(value.get("text", ""))
	parts = []
	if "content" in value:
		content = jira_value_to_text(value["content"])
		if content:
			parts.append(content)
	if "text" in value and value.get("type") != "text":
		parts.append(str(value["text"]))
	return "\n".join(parts).strip()


def jira_session(config: dict[str, str]) -> requests.Session:
	session = requests.Session()
	session.auth = (config["jira_email"], config["jira_api_token"])
	session.headers.update({
		"Accept": "application/json",
		"Content-Type": "application/json",
	})
	return session


def fetch_jira_issue(issue_key: str, config: dict[str, str]) -> dict[str, str]:
	url = f"{config['jira_base_url']}/rest/api/3/issue/{issue_key}"
	with jira_session(config) as session:
		response = session.get(
			url,
			params={"fields": "summary,description"},
			timeout=60,
		)
		response.raise_for_status()
	issue = response.json()
	fields = issue.get("fields", {})
	title = jira_value_to_text(fields.get("summary"))
	description = jira_value_to_text(fields.get("description"))
	if not title:
		raise ValueError(f"Jira issue {issue_key} has no title.")
	embedded_text = f"Title: {title}\n\nDescription: {description}".strip()
	return {
		"defect_id": issue_key,
		"title": title,
		"description": description,
		"embedded_text": embedded_text,
	}


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> SentenceTransformer:
	return SentenceTransformer(model_name)


def extract_document_title(document: str) -> str:
	if document.startswith("Title:"):
		title = document.split("\n\nDescription:", 1)[0]
		return title.removeprefix("Title:").strip()
	return ""


def search_similar_defects(
	text: str,
	db_dir: Path,
	collection_name: str,
	embedding_model_name: str,
	progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
	client = chromadb.PersistentClient(path=str(db_dir))
	try:
		collection = client.get_collection(name=collection_name)
	except Exception as error:
		raise RuntimeError(
			f"Chroma collection '{collection_name}' was not found. "
			"Run jira-defect-embedding.py first."
		) from error
	if collection.count() == 0:
		raise RuntimeError(f"Chroma collection '{collection_name}' is empty.")

	model = load_embedding_model(embedding_model_name)
	query_embedding = model.encode(text, normalize_embeddings=True).tolist()
	if progress_callback:
		progress_callback(
			f"Embedding complete ({len(query_embedding)} dimensions) using "
			f"'{embedding_model_name}'."
		)
	result_count = min(TOP_K, collection.count())
	results = collection.query(
		query_embeddings=[query_embedding],
		n_results=result_count,
		include=["documents", "metadatas", "distances"],
	)

	matches = []
	for record_id, document, metadata, distance in zip(
		results["ids"][0],
		results["documents"][0],
		results["metadatas"][0],
		results["distances"][0],
	):
		matches.append({
			"defect_id": record_id,
			"title": extract_document_title(document),
			"document": document,
			"root_cause": (metadata or {}).get("root_cause", ""),
			"similarity": max(0.0, min(1.0, 1.0 - float(distance))),
		})
	if progress_callback:
		progress_callback(
			f"Similarity search complete. Retrieved {len(matches)} of "
			f"{collection.count()} indexed defects."
		)
	return matches


def build_gemini_prompt(
	issue: dict[str, str],
	matches: list[dict[str, Any]],
) -> str:
	historical_context = []
	for index, match in enumerate(matches, start=1):
		historical_context.append(
			f"Similar defect {index}:\n"
			f"ID: {match['defect_id']}\n"
			f"Similarity: {match['similarity']:.4f}\n"
			f"Defect: {match['document']}\n"
			f"Known root cause: {match['root_cause'] or 'Not available'}"
		)
	return (
		"You are an expert software defect triage assistant.\n\n"
		"Analyze the current Jira defect using the historical defects below. "
		"Do not invent facts that are not supported by the available evidence.\n\n"
		f"Current defect ({issue['defect_id']}):\n{issue['embedded_text']}\n\n"
		"Historical matches:\n" + "\n\n".join(historical_context) + "\n\n"
		"Provide a concise recommendation with these sections:\n"
		"1. Likely root cause\n"
		"2. Evidence from similar defects\n"
		"3. Recommended investigation steps\n"
		"4. Confidence level"
	)


def generate_recommendation(
	prompt: str,
	config: dict[str, str],
) -> str:
	client = genai.Client(api_key=config["gemini_api_key"])
	response = client.models.generate_content(
		model=config["gemini_model"],
		contents=prompt,
	)
	recommendation = (response.text or "").strip()
	if not recommendation:
		raise RuntimeError("Gemini returned an empty recommendation.")
	return recommendation


def post_jira_comment(
	issue_key: str,
	recommendation: str,
	matches: list[dict[str, Any]],
	config: dict[str, str],
) -> None:
	match_lines = "\n".join(
		f"- {match['defect_id']} (similarity {match['similarity']:.4f})"
		for match in matches
	)
	comment_text = (
		"AI Defect Triage Recommendation\n\n"
		f"{recommendation}\n\n"
		"Similar defects used:\n"
		f"{match_lines}\n\n"
		"Generated by the Jira defect triage application."
	)
	payload = {
		"body": {
			"type": "doc",
			"version": 1,
			"content": [{
				"type": "paragraph",
				"content": [{"type": "text", "text": comment_text}],
			}],
		},
	}
	url = f"{config['jira_base_url']}/rest/api/3/issue/{issue_key}/comment"
	with jira_session(config) as session:
		response = session.post(url, json=payload, timeout=60)
		response.raise_for_status()


def render_match(match: dict[str, Any], index: int) -> None:
	st.markdown(f"#### {index}. {match['defect_id']}")
	st.metric("Similarity", f"{match['similarity']:.2%}")
	st.text(match["document"])
	if match["root_cause"]:
		st.caption(f"Known root cause: {match['root_cause']}")


def run_triage(issue_key: str) -> None:
	config = get_config()
	with st.status("Running defect triage...", expanded=True) as status:
		st.write("Fetching Jira issue...")
		issue = fetch_jira_issue(issue_key, config)
		st.write(
			f"Jira fetch complete: **{issue['title']}** "
			f"({len(issue['description'])} description characters)."
		)
		st.json({
			"jira_id": issue["defect_id"],
			"title": issue["title"],
			"description_length": len(issue["description"]),
		})
		st.write("Creating embedding and searching ChromaDB...")
		matches = search_similar_defects(
			issue["embedded_text"],
			DEFAULT_DB_DIR,
			DEFAULT_COLLECTION,
			DEFAULT_EMBEDDING_MODEL,
			progress_callback=st.write,
		)
		st.dataframe(
			[
				{
					"rank": index,
					"defect_id": match["defect_id"],
					"title": match["title"],
					"similarity": f"{match['similarity']:.2%}",
				}
				for index, match in enumerate(matches, start=1)
			],
				hide_index=True,
			)
		st.write("Generating Gemini recommendation...")
		recommendation = generate_recommendation(
			build_gemini_prompt(issue, matches),
			config,
		)
		st.write("Gemini recommendation complete.")
		st.markdown(recommendation)
		st.write("Posting recommendation to Jira...")
		post_jira_comment(issue_key, recommendation, matches, config)
		st.write(f"Jira comment posted successfully to **{issue_key}**.")
		status.update(label="Triage completed", state="complete")

	st.subheader(f"Current Jira defect: {issue['defect_id']}")
	st.markdown(f"**{issue['title']}**")
	st.text(issue["description"] or "No description provided.")

	st.subheader("Top similar defects")
	for index, match in enumerate(matches, start=1):
		render_match(match, index)

	st.subheader("Gemini recommendation")
	st.markdown(recommendation)
	st.success(f"Recommendation posted to Jira issue {issue_key}.")


def main() -> None:
	st.set_page_config(
		page_title="Defect Triage - RAG Agent",
		page_icon="JT",
		layout="wide",
	)
	st.title("Defect Triage RAG Agent")
	st.caption("Retrieve, compare, explain, and annotate a Jira defect.")

	with st.form("triage_form"):
		issue_key = st.text_input("Jira issue key", placeholder="SCRUM-66")
		submitted = st.form_submit_button("Analyze defect", type="primary")

	if submitted:
		issue_key = issue_key.strip().upper()
		if not re.fullmatch(r"[A-Z][A-Z0-9_]*-\d+", issue_key):
			st.error("Enter a valid Jira issue key, for example SCRUM-66.")
			return
		try:
			run_triage(issue_key)
		except requests.HTTPError as error:
			status_code = error.response.status_code if error.response is not None else "unknown"
			error_msg = ""
			if error.response is not None:
				try:
					data = error.response.json()
					if "errorMessages" in data and data["errorMessages"]:
						error_msg = ": " + " ".join(data["errorMessages"])
				except Exception:
					error_msg = f": {error.response.text}"
			st.error(f"Jira request failed with HTTP {status_code}{error_msg}")
		except Exception as error:
			st.error(f"Triage failed: {error}")


if __name__ == "__main__":
	main()
