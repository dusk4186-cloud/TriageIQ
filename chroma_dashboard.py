"""Streamlit dashboard for inspecting a persistent ChromaDB database."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import streamlit as st


st.set_page_config(page_title="ChromaDB Inspector", page_icon="DB", layout="wide")


def directory_size(path: Path) -> int:
	return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def format_bytes(size: int) -> str:
	units = ["B", "KB", "MB", "GB"]
	value = float(size)
	for unit in units:
		if value < 1024 or unit == units[-1]:
			return f"{value:.2f} {unit}"
		value /= 1024
	return f"{size} B"


def get_collection_data(collection: Any) -> dict[str, Any]:
	count = collection.count()
	data: dict[str, Any] = {
		"ids": [],
		"documents": [],
		"metadatas": [],
		"embeddings": [],
	}
	if count:
		data = collection.get(include=["documents", "metadatas", "embeddings"])

	embeddings = data.get("embeddings")
	if embeddings is None:
		embeddings = []
	metadata = data.get("metadatas") or []
	embedding_dimension = len(embeddings[0]) if len(embeddings) else 0
	metadata_keys = sorted({key for item in metadata if item for key in item})
	return {
		"count": count,
		"data": data,
		"embedding_dimension": embedding_dimension,
		"metadata_keys": metadata_keys,
	}


def metadata_frame(data: dict[str, Any]) -> pd.DataFrame:
	rows = []
	for index, record_id in enumerate(data.get("ids") or []):
		row = {
			"id": record_id,
			"document": (data.get("documents") or [""])[index],
		}
		row.update((data.get("metadatas") or [{}])[index] or {})
		rows.append(row)
	return pd.DataFrame(rows)


def render_collection(collection: Any) -> None:
	info = get_collection_data(collection)
	data = info["data"]

	count_col, dimension_col, metadata_col, space_col = st.columns(4)
	count_col.metric("Embeddings", f"{info['count']:,}")
	dimension_col.metric("Vector dimensions", f"{info['embedding_dimension']:,}")
	metadata_col.metric("Metadata fields", f"{len(info['metadata_keys']):,}")
	space_col.metric("Distance space", collection.metadata.get("hnsw:space", "default"))

	st.subheader("Collection metadata")
	st.json(collection.metadata or {})

	if not info["count"]:
		st.info("This collection is empty.")
		return

	st.subheader("Records and metadata")
	frame = metadata_frame(data)
	search = st.text_input("Filter records", key=f"filter_{collection.name}")
	if search:
		mask = frame.astype(str).apply(
			lambda column: column.str.contains(search, case=False, na=False)
		).any(axis=1)
		frame = frame[mask]
	st.dataframe(frame, use_container_width=True, hide_index=True)

	st.subheader("Embedding vectors")
	embeddings = data.get("embeddings")
	if embeddings is None:
		embeddings = []
	embedding_rows = []
	for record_id, vector in zip(data.get("ids") or [], embeddings):
		embedding_rows.append({
			"id": record_id,
			"dimension": len(vector),
			"first_values": str([round(value, 6) for value in vector[:8]]),
		})
	st.dataframe(pd.DataFrame(embedding_rows), use_container_width=True, hide_index=True)

	with st.expander("View full documents"):
		for record_id, document in zip(data.get("ids") or [], data.get("documents") or []):
			st.markdown(f"**{record_id}**")
			st.text(document)


def main(db_dir: Path) -> None:
	st.title("ChromaDB Inspector")
	st.caption(f"Persistent database: `{db_dir.resolve()}`")

	if not db_dir.exists():
		st.error(f"Database directory does not exist: {db_dir}")
		st.stop()

	client = chromadb.PersistentClient(path=str(db_dir))
	collections = client.list_collections()
	collection_names = [collection.name for collection in collections]
	db_size = directory_size(db_dir)
	total_records = sum(client.get_collection(name=name).count() for name in collection_names)

	size_col, collection_col, record_col = st.columns(3)
	size_col.metric("Database size", format_bytes(db_size))
	collection_col.metric("Collections", f"{len(collection_names):,}")
	record_col.metric("Total records", f"{total_records:,}")

	if not collections:
		st.info("No collections found in this database.")
		return

	st.subheader("Collections")
	collection_rows = []
	for collection in collections:
		info = get_collection_data(collection)
		collection_rows.append({
			"name": collection.name,
			"records": info["count"],
			"dimensions": info["embedding_dimension"],
			"metadata fields": ", ".join(info["metadata_keys"]) or "none",
			"distance space": collection.metadata.get("hnsw:space", "default"),
		})
	st.dataframe(pd.DataFrame(collection_rows), use_container_width=True, hide_index=True)

	selected_name = st.selectbox("Inspect collection", collection_names)
	selected = client.get_collection(name=selected_name)
	render_collection(selected)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Inspect a persistent ChromaDB database")
	parser.add_argument("--db-dir", type=Path, default=Path("./chroma_db"))
	args = parser.parse_args()
	main(args.db_dir)
