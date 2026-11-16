# Jira Defect Triage Streamlit App

## Purpose

`Defect_Triaging.py` is a Streamlit application that helps a developer triage one Jira defect using historical defects stored in ChromaDB and a Gemini-generated recommendation.

The app performs retrieval-augmented triage:

1. Fetch the selected Jira issue.
2. Convert its title and description into an embedding.
3. Retrieve the three most similar historical defects.
4. Build a prompt containing the current defect and retrieved evidence.
5. Generate a recommendation with Gemini.
6. Post the recommendation back to the Jira issue as a comment.

## Functional flow

```text
User enters Jira key
        |
        v
Validate key format
        |
        v
Fetch title and description from Jira
        |
        v
Embed current defect with SentenceTransformer
        |
        v
Search ChromaDB collection: jira_defects
        |
        v
Retrieve top 3 similar defects
        |
        v
Generate Gemini triage recommendation
        |
        v
Display results in Streamlit
        |
        v
Post recommendation to Jira
```

## Requirements

Activate the project virtual environment and install the dependencies:

```powershell
python -m pip install streamlit chromadb requests python-dotenv google-genai sentence-transformers
```

The app expects the ChromaDB index to have already been created by `jira-defect-embedding.py`.

## Configuration

Create a `.env` file in the directory from which you run the app:

```dotenv
JIRA_BASE_URL=https://your-company.atlassian.net
JIRA_EMAIL=developer@example.com
JIRA_API_TOKEN=your-jira-api-token
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.1-pro-preview
```

Required values:

- `JIRA_BASE_URL`: Jira site URL without a trailing slash
- `JIRA_EMAIL`: Jira account email
- `JIRA_API_TOKEN`: Jira API token
- `GEMINI_API_KEY`: Google Gemini API key

`GEMINI_MODEL` is optional. If omitted, the application uses `gemini-3.1-pro-preview`.

Do not commit `.env`, Jira tokens, or Gemini keys to source control.

## Prepare the ChromaDB index

Run the indexing script first from this directory:

```powershell
python jira-defect-embedding.py
```

For local JSON input instead of the Jira API:

```powershell
python jira-defect-embedding.py --input-json .\defectdump.json
```

The triage app expects these defaults:

- Database directory: `./chroma_db`
- Collection: `jira_defects`
- Embedding model: `all-MiniLM-L6-v2`

The indexing and triage scripts must use the same database directory, collection name, and embedding model.

## Start the Streamlit app

From the `Defect Triage` directory:

```powershell
python -m streamlit run Defect_Triaging.py
```

Then enter a Jira issue key such as:

```text
PAY-1842
```

The app accepts issue keys matching this pattern:

```text
PROJECT-123
```

## What the app displays

During triage, the status panel shows:

- Jira fetch progress
- Current issue title and description length
- Embedding progress
- Similarity search progress
- Retrieved defect summary table
- Gemini generation progress
- Jira comment posting status

After completion, the page displays:

- Current Jira defect
- Top three similar defects
- Similarity values
- Historical root causes when available
- Gemini recommendation
- Success confirmation after Jira posting

## Recommendation format

The Gemini prompt asks for four sections:

1. Likely root cause
2. Evidence from similar defects
3. Recommended investigation steps
4. Confidence level

The recommendation and the IDs of the similar defects are included in the Jira comment.

## Error handling

The app reports errors in the Streamlit page for cases such as:

- Invalid Jira issue key format
- Missing `.env` settings
- Jira HTTP failures
- Missing or empty ChromaDB collection
- Missing embedding model
- Empty Gemini response
- Failed Jira comment request

## Troubleshooting

### Collection not found

Run `jira-defect-embedding.py` first. If the collection still cannot be found, verify that both scripts are being run from the same directory and that `./chroma_db` exists.

### No similar defects returned

Check that the collection contains records and that the index was created with the same embedding model:

```text
all-MiniLM-L6-v2
```

### Jira request fails

Check the Jira URL, email, API token, issue permissions, and whether the issue key exists. The account must be allowed to read the issue and add comments.

### Gemini request fails

Check `GEMINI_API_KEY`, the selected model name, network access, and the API quota for the Google account.

### Relative path confusion

The defaults use paths relative to the current working directory, not necessarily the script location. Start both programs from the `Defect Triage` directory, or adjust the scripts if you need project-root paths.

## Security notes

- Keep API keys in environment variables or a local `.env` file.
- Never print tokens in logs or Streamlit output.
- Use a Jira service account with only the permissions required for reading defects and posting comments.
- Review generated recommendations before using them for production decisions.
- Treat retrieved defect text as untrusted input when extending the prompt or adding automated actions.
