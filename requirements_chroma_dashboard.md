# Requirement Prompt: ChromaDB Dashboard

## Project Title
Build a Streamlit dashboard to inspect a persistent ChromaDB database.

## Objective
Create an interactive user interface that helps users view database details, collection records, metadata, and embedding information without writing database queries manually.

## Problem Statement
Your project stores vector embeddings in ChromaDB, but it is difficult to inspect the database visually. You need to build a dashboard that shows the database structure in a simple and understandable format.

## Required Features
Students must build a program that:

1. Connects to a persistent ChromaDB database.
2. Loads all collections from the database.
3. Displays database size and record count.
4. Shows collection names and metadata.
5. Allows the user to select a collection to inspect.
6. Displays stored records and metadata in a table.
7. Shows vector dimensions and sample embedding values.
8. Adds a text filter to search through records.
9. Uses Streamlit to create the dashboard UI.

## Functional Requirements
- Use Python.
- Use `streamlit` for the dashboard interface.
- Use `chromadb` for data access.
- Use `pandas` for tabular display.
- Accept a database path using command-line or default value.
- Show helpful messages when the database is missing or empty.

## Suggested Program Workflow
1. Connect to the ChromaDB directory.
2. List all available collections.
3. Calculate total database size and total record count.
4. Show summary cards for the database.
5. Let the user select a collection.
6. Fetch all records and metadata for the selected collection.
7. Display the records in a table.
8. Add a search box for filtering data.
9. Show embedding summary information.

## Deliverables
Submit:
- one Streamlit Python file
- a README file with instructions to run the app
- examples of how to inspect the database

## Evaluation Criteria
Your work will be evaluated on:
- correct ChromaDB connection
- clear dashboard layout
- data display quality
- filtering and inspection functions
- proper metadata and embedding presentation
- easy execution for students

## Example UI Features
- summary metrics cards
- database overview table
- collection selector dropdown
- record metadata viewer
- embedding dimension display
- search/filter input

## Bonus Task
Add a feature that lets the user view full document text for each stored item in a collapsible section.
