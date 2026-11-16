# ChromaDB Dashboard

## What this program does
This script creates a Streamlit dashboard to inspect and explore a persistent ChromaDB database.

It allows users to view database size, collection names, record counts, metadata fields, and stored embeddings without writing custom database queries.

## Files involved
- `chroma_dashboard.py` – main dashboard file
- `chroma_db/` – database folder created by the other scripts

## Prerequisites
Install the required libraries:

```bash
pip install streamlit pandas chromadb
```

## How to run
Open a terminal in the project folder and run:

```bash
streamlit run chroma_dashboard.py -- --db-dir ./chroma_db
```

This starts the dashboard in your browser.

If Streamlit is not installed globally, you can also run it through Python:

```bash
python -m streamlit run chroma_dashboard.py -- --db-dir ./chroma_db
```

## What you will see in the dashboard
Once the app opens:
- total database size
- number of collections
- total record count
- metadata for each collection
- selection dropdown to inspect a collection
- table of stored records and metadata
- vector dimension information
- search filter to filter records by text

## How to use the dashboard
1. Start the app.
2. Choose the database path if needed.
3. Select a collection from the dropdown.
4. View metadata and embeddings for each stored record.
5. Use the text filter to search inside the documents or metadata.

## Expected output
The app shows a visual interface in the browser rather than plain terminal text. It helps users understand how data is structured inside ChromaDB.

## Notes for users
- This dashboard is useful for debugging vector databases.
- It helps visualize whether embeddings were inserted correctly.
- You can inspect the records, metadata fields, and vector lengths in an interactive UI.
- It is particularly helpful when you want to verify the output of the indexing scripts before building a search application.
