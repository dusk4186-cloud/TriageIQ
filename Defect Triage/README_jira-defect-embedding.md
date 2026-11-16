# Jira Defect Embedding and ChromaDB Indexing

## Purpose

`jira-defect-embedding.py` prepares historical Jira defects for semantic search. It can either:

- Fetch Jira bugs through the Jira REST API.
- Read a local Jira-export JSON file without calling Jira.

The script normalizes each defect, writes a cleaned JSON file, creates sentence embeddings with `all-MiniLM-L6-v2`, and stores the vectors in a persistent ChromaDB collection.

This script should normally be run before `Defect_Triaging.py`.

## Pipeline

```text
Jira REST API or local JSON
          |
          v
Normalize title, description, comments, labels
          |
          v
Extract root cause and log metadata
          |
          v
Write jira_defects.json
          |
          v
Create sentence embeddings
          |
          v
Upsert vectors into ChromaDB
```

## Requirements

Activate the project virtual environment and install the dependencies:

```powershell
python -m pip install chromadb requests python-dotenv sentence-transformers
```

The script uses these Python packages:

- `chromadb` for persistent vector storage
- `requests` for Jira REST API calls
- `python-dotenv` for `.env` configuration
- `sentence-transformers` for embeddings

## Configuration for Jira API mode

Create a `.env` file in the directory from which you run the script:

```dotenv
JIRA_BASE_URL=https://your-company.atlassian.net
JIRA_EMAIL=developer@example.com
JIRA_API_TOKEN=your-jira-api-token
JIRA_JQL=issuetype = Bug ORDER BY key
```

Use a Jira API token rather than a Jira account password. Do not commit `.env` or API tokens to source control.

The default query is:

```text
issuetype = Bug ORDER BY key
```

## Run with Jira

Run this command from the `Defect Triage` directory:

```powershell
python jira-defect-embedding.py
```

This creates or updates:

- `./jira_defects.json`
- `./chroma_db/`
- ChromaDB collection `jira_defects`

The script uses Jira pagination and requests up to 100 records per page until all matching issues are loaded.

Override the JQL query when needed:

```powershell
python jira-defect-embedding.py --jql "project = PAY AND issuetype = Bug ORDER BY created DESC"
```

## Run with a local JSON export

Use `--input-json` to avoid a Jira API call:

```powershell
python jira-defect-embedding.py --input-json .\jira_defects.json
```

You can also use a different input and output file:

```powershell
python jira-defect-embedding.py `
  --input-json .\defectdump.json `
  --json-output .\normalized_defects.json
```

The input JSON must contain a list of Jira-style issue objects. The normalizer accepts either Jira field names or simplified fields such as `title` and `description`.

## Optional arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--db-dir` | `./chroma_db` | Persistent ChromaDB directory |
| `--collection` | `jira_defects` | ChromaDB collection name |
| `--json-output` | `./jira_defects.json` | Normalized JSON output path |
| `--model` | `all-MiniLM-L6-v2` | Sentence-transformer model name |
| `--input-json` | Not set | Use local JSON instead of Jira |
| `--jql` | `JIRA_JQL` or default | Jira query override |

Example with explicit paths:

```powershell
python jira-defect-embedding.py `
  --db-dir .\chroma_db `
  --collection jira_defects `
  --json-output .\jira_defects.json `
  --model all-MiniLM-L6-v2
```

## Data extracted

Each normalized record contains:

- `defect_id`: Jira key, issue ID, or generated `local-N` identifier
- `embedded_text`: title and description used for embedding
- `root_cause`: values found in comments or labels using `Root cause:` or similar prefixes
- `log`: values found in comments or labels using `Log:` or `Logs:` prefixes

The ChromaDB document contains `embedded_text`. The current implementation stores `root_cause` as metadata.

## Expected output

The terminal reports:

- The normalized JSON output path
- Number of indexed defects
- Embedding dimensions
- ChromaDB database path
- Collection name

A first run may download the sentence-transformer model and can take longer than later runs.

## Troubleshooting

### Missing Jira configuration

Set `JIRA_BASE_URL`, `JIRA_EMAIL`, and `JIRA_API_TOKEN` in `.env`, or use `--input-json` for local-only processing.

### Empty or invalid JSON

The input file must contain a JSON list. Check that each record has a key or ID and a title or summary.

### Existing collection contains old records

The script uses `upsert`, so records with matching IDs are updated. It does not automatically delete defects that were removed from the source data. Use a separate collection or rebuild the database when a clean index is required.

### Triage cannot find the collection

Run the script from the same working directory used by `Defect_Triaging.py`, or pass matching database and collection settings in both programs. Their defaults are `./chroma_db` and `jira_defects`.
