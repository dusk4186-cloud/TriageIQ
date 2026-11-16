"""Fetch Jira defects, prepare embedding JSON, and index them in ChromaDB."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


DEFAULT_DB_DIR = Path("./chroma_db")
DEFAULT_COLLECTION = "jira_defects"
DEFAULT_JSON_OUTPUT = Path("./jira_defects.json")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_JQL = "issuetype = Bug ORDER BY key"
PAGE_SIZE = 100


@dataclass(frozen=True)
class JiraConfig:
	base_url: str
	email: str
	api_token: str
	jql: str


def load_config(jql_override: str | None = None) -> JiraConfig:
	load_dotenv()
	values = {
		"JIRA_BASE_URL": os.getenv("JIRA_BASE_URL", "").strip().rstrip("/"),
		"JIRA_EMAIL": os.getenv("JIRA_EMAIL", "").strip(),
		"JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN", "").strip(),
	}
	missing = [name for name, value in values.items() if not value]
	if missing:
		raise RuntimeError(
			"Missing Jira configuration: " + ", ".join(missing) +
			". Set these values in .env or the environment."
		)
	return JiraConfig(
		base_url=values["JIRA_BASE_URL"],
		email=values["JIRA_EMAIL"],
		api_token=values["JIRA_API_TOKEN"],
		jql=jql_override or os.getenv("JIRA_JQL", DEFAULT_JQL),
	)


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


def extract_prefixed_values(texts: list[str], prefix_pattern: str) -> list[str]:
	values = []
	pattern = re.compile(prefix_pattern, re.IGNORECASE)
	for text in texts:
		for line in text.splitlines():
			match = pattern.match(line.strip())
			if match and match.group(1).strip():
				values.append(match.group(1).strip())
	return values


def extract_comment_fields(
	comments: list[str],
	labels: list[str],
	description: str = "",
) -> tuple[str, str]:
	all_text = comments + labels + ([description] if description else [])
	root_causes = extract_prefixed_values(
		all_text,
		r"^(?:\*\*root[ _-]?cause\*\*|root[ _-]?cause)\s*:\s*(.+)$",
	)
	logs = extract_prefixed_values(
		all_text,
		r"^(?:\*\*logs?\*\*|logs?)\s*:\s*(.+)$",
	)
	return "\n".join(root_causes), "\n".join(logs)


def normalize_defect(issue: dict[str, Any]) -> dict[str, str]:
	fields = issue.get("fields", issue)
	defect_id = str(issue.get("key") or issue.get("id") or fields.get("defect_id", ""))
	title = jira_value_to_text(fields.get("summary") or fields.get("title"))
	description = jira_value_to_text(fields.get("description"))
	comments_data = fields.get("comment", [])
	if isinstance(comments_data, dict):
		comments_data = comments_data.get("comments", [])
	comments = [
		jira_value_to_text(comment.get("body", comment))
		for comment in comments_data
	]
	labels = [str(label) for label in fields.get("labels", [])]
	root_cause, log = extract_comment_fields(comments, labels, description)
	embedded_text = f"Title: {title}\n\nDescription: {description}".strip()
	return {
		"defect_id": defect_id,
		"embedded_text": embedded_text,
		"root_cause": root_cause,
		"log": log,
	}


def fetch_all_defects(config: JiraConfig) -> list[dict[str, Any]]:
	url = f"{config.base_url}/rest/api/3/search/jql"
	issues = []
	next_page_token = None
	with requests.Session() as session:
		session.auth = (config.email, config.api_token)
		session.headers.update({
			"Accept": "application/json",
			"Content-Type": "application/json",
		})
		while True:
			payload = {
				"jql": config.jql,
				"maxResults": PAGE_SIZE,
				"fields": ["summary", "description", "comment", "labels"],
			}
			if next_page_token:
				payload["nextPageToken"] = next_page_token
			response = session.post(
				url,
				json=payload,
				timeout=60,
			)
			response.raise_for_status()
			page = response.json()
			page_issues = page.get("issues", [])
			issues.extend(page_issues)
			next_page_token = page.get("nextPageToken")
			if page.get("isLast", True) or not next_page_token or not page_issues:
				return issues


def load_input_records(input_path: Path) -> list[dict[str, str]]:
	data = json.loads(input_path.read_text(encoding="utf-8"))
	if not isinstance(data, list):
		raise ValueError(f"Input JSON must contain a list: {input_path}")
	records = []
	for index, item in enumerate(data, start=1):
		record = normalize_defect(item)
		if not record["defect_id"]:
			record["defect_id"] = f"local-{index}"
		records.append(record)
	return records


def write_json(records: list[dict[str, str]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(records, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)
	print(f"Embedding JSON written to {output_path.resolve()}")


def save_vectors(
	records: list[dict[str, str]],
	db_dir: Path,
	collection_name: str,
	model_name: str,
) -> None:
	if not records:
		print("No defects found; nothing to index.")
		return
	model = SentenceTransformer(model_name)
	embeddings = model.encode(
		[record["embedded_text"] for record in records],
		normalize_embeddings=True,
	)
	client = chromadb.PersistentClient(path=str(db_dir))
	collection = client.get_or_create_collection(
		name=collection_name,
		metadata={"hnsw:space": "cosine"},
	)
	collection.upsert(
		ids=[record["defect_id"] for record in records],
		documents=[record["embedded_text"] for record in records],
		embeddings=embeddings.tolist(),
		metadatas=[{"root_cause": record["root_cause"]} for record in records],
	)
	print(f"Indexed {len(records)} defects in Chroma collection '{collection_name}'")
	print(f"Embedding dimensions: {len(embeddings[0])}")
	print(f"Database path: {db_dir.resolve()}")


def main(
	db_dir: Path,
	collection_name: str,
	json_output: Path,
	model_name: str,
	input_json: Path | None,
	jql: str | None,
) -> None:
	if input_json:
		records = load_input_records(input_json)
	else:
		config = load_config(jql)
		records = [normalize_defect(issue) for issue in fetch_all_defects(config)]
	write_json(records, json_output)
	save_vectors(records, db_dir, collection_name, model_name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Fetch Jira defects, create embedding JSON, and index ChromaDB"
	)
	parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
	parser.add_argument("--collection", default=DEFAULT_COLLECTION)
	parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
	parser.add_argument("--model", default=DEFAULT_MODEL)
	parser.add_argument(
		"--input-json",
		type=Path,
		help="Use a local Jira export instead of calling the Jira API",
	)
	parser.add_argument("--jql", help="Override JIRA_JQL for the Jira query")
	args = parser.parse_args()
	main(
		args.db_dir,
		args.collection,
		args.json_output,
		args.model,
		args.input_json,
		args.jql,
	)
