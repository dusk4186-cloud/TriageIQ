@echo off
setlocal

cd /d "%~dp0"

echo Starting Smart Jira MCP Server (Tools + Defect Triage RAG)...
python server.py
