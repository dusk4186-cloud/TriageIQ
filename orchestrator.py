"""
Jira AI Orchestrator with Intelligent Multi-Model Failover & ChromaDB RAG
-------------------------------------------------------------------------
Features:
1. Natural-language Jira interactions via Smart Jira MCP Server tools.
2. Defect Intelligence & RAG: Automatically searches historical ChromaDB defects,
   analyzes logs, compares root causes, and triages bugs.
3. Multi-Model Cascade: Switches to backup models if a model is busy/overloaded (503) or rate-limited.
4. Multi-Key Rotation: Switches to backup Gemini API keys if quota (429) is hit.
5. Groq LLM Cross-Provider Fallback: Pre-wired slot for Groq (Llama 3.3 70B) when configured.
6. Context Preservation: Preserves full conversation history across model failovers.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Safe Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx
from google import genai
from google.genai import types, errors
from mcp_client import JiraMCPClient, MCPError

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for _log_name in (
    "httpx", "httpx2", "httpcore", "httpcore2",
    "google", "google.genai", "google_genai",
    "urllib3", "huggingface_hub", "sentence_transformers"
):
    logging.getLogger(_log_name).setLevel(logging.WARNING)

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)
load_dotenv(BASE_DIR / "Defect Triage" / ".env")


# ============================================================================
# Schema Sanitizer for Gemini Function Calling
# ============================================================================

def sanitize_schema(schema: dict) -> dict:
    """Converts and sanitizes an MCP JSON schema into a Gemini-compliant format."""
    if not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}

    raw_type = schema.get("type", "OBJECT")
    if isinstance(raw_type, list):
        raw_type = raw_type[0]

    type_map = {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
        "any": "STRING",
    }
    stype = type_map.get(str(raw_type).lower(), "STRING")
    clean = {"type": stype}

    if "description" in schema:
        clean["description"] = str(schema["description"])[:500]

    if "enum" in schema and isinstance(schema["enum"], list):
        clean["enum"] = [str(x) for x in schema["enum"]]

    if stype == "ARRAY":
        items = schema.get("items")
        clean["items"] = sanitize_schema(items) if isinstance(items, dict) else {"type": "STRING"}

    if stype == "OBJECT":
        props = schema.get("properties", {})
        clean_props = {}
        if isinstance(props, dict):
            for k, v in props.items():
                clean_props[k] = sanitize_schema(v) if isinstance(v, dict) else {"type": "STRING"}
        clean["properties"] = clean_props

        raw_req = schema.get("required", [])
        if isinstance(raw_req, list):
            valid_req = [r for r in raw_req if r in clean_props]
            if valid_req:
                clean["required"] = valid_req

    return clean


# ============================================================================
# System Instructions for Jira & Defect Triage SRE
# ============================================================================

def format_tool_result_summary(tool_name: str, payload: Any) -> str:
    """Formats a concise system message for tool execution results."""
    parsed = payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            pass

    if isinstance(parsed, dict):
        # 1. Read-only guardrail blocked
        if (
            parsed.get("status") == "blocked"
            or parsed.get("read_only") is True
            or ("blocked" in str(parsed.get("error", "")).lower() and "read-only" in str(parsed.get("error", "")).lower())
        ):
            return "[Tool result] Write operation blocked: read-only mode"

        # 2. General error
        if "error" in parsed:
            err_msg = str(parsed["error"]).split("\n")[0]
            if len(err_msg) > 90:
                err_msg = err_msg[:87] + "..."
            return f"[Tool result] Error: {err_msg}"

        # 3. Defect similarity search
        if tool_name == "jira_find_similar_defects":
            matches = parsed.get("matches", []) if isinstance(parsed.get("matches"), list) else []
            return f"[Tool result] Found {len(matches)} similar historical defect(s)"

        # 4. Defect triage
        if tool_name == "jira_triage_defect":
            key = parsed.get("issue_key") or "defect"
            return f"[Tool result] Triage completed for {key}"

        # 5. Get issue
        if tool_name == "jira_get_issue":
            key = parsed.get("key") or "issue"
            status = parsed.get("fields", {}).get("status", {}).get("name") if isinstance(parsed.get("fields"), dict) else ""
            status_str = f" ({status})" if status else ""
            return f"[Tool result] Retrieved {key}{status_str}"

        # 6. Search issues
        if tool_name == "jira_search_issues":
            issues = parsed.get("issues", [])
            total = parsed.get("total", len(issues))
            return f"[Tool result] Retrieved {total} issue(s)"

        # 7. Add comment
        if tool_name == "jira_add_comment":
            return f"[Tool result] Comment added to {parsed.get('issue_key', 'issue')}"

        # 8. Create issue
        if tool_name == "jira_create_issue":
            return f"[Tool result] Created issue {parsed.get('key', '')}"

        # 9. Transition issue
        if tool_name == "jira_transition_issue":
            return f"[Tool result] Transition applied successfully"

        # 10. List projects
        if tool_name == "jira_list_projects":
            projects = parsed.get("projects", [])
            return f"[Tool result] Found {len(projects)} project(s)"

        return "[Tool result] Success"

    elif isinstance(parsed, list):
        return f"[Tool result] Returned {len(parsed)} item(s)"

    # Fallback for plain text
    text_lower = str(parsed).lower()
    if "blocked" in text_lower and "read-only" in text_lower:
        return "[Tool result] Write operation blocked: read-only mode"

    return "[Tool result] Success"


SYSTEM_INSTRUCTION = (
    "You are an expert Jira AI Assistant and Site Reliability Engineer with direct access to Jira MCP tools "
    "and a ChromaDB vector database containing historical defect embeddings.\n"
    "Default project key: 'SCRUM' (unless the user specifies another key).\n\n"
    "Core Capabilities:\n"
    "1. Jira Issue Management: Search issues (JQL or free text), inspect full details, create tickets, "
    "update fields, add comments, and transition workflow states (e.g. 'To Do' -> 'In Progress' -> 'Done').\n"
    "2. Defect Intelligence & Historical RAG: When a user asks 'have we seen this before?', describes an error, "
    "or pastes stack traces, invoke `jira_find_similar_defects` to query ChromaDB for past related defects, "
    "similarity percentages, and known root causes.\n"
    "3. Automated Triage: When asked to triage an issue or investigate a ticket, invoke `jira_triage_defect`. "
    "Only set post_comment_to_jira=True if the user explicitly asks to post the triage recommendation to Jira.\n\n"
    "Response Guidelines:\n"
    "- When displaying issues, format them cleanly in markdown with Key, Summary, Status, Priority, and Root Cause.\n"
    "- If a user asks to search for their assigned tickets, query `project = SCRUM ORDER BY updated DESC` or use text search.\n"
    "- Be concise, proactive, and actionable."
)


# ============================================================================
# Gemini Multi-Model & Multi-Key Failover Session
# ============================================================================

class GeminiOrchestrator:
    def __init__(self, keys: List[str], models: List[str], tools: list):
        self.keys = keys
        self.models = models
        self.tools = tools
        self.current_key_idx = 0
        self.current_model_idx = 0
        self.client: Optional[genai.Client] = None
        self.chat = None
        self._init_chat()

    @property
    def current_model(self) -> str:
        return self.models[self.current_model_idx]

    @property
    def current_key_preview(self) -> str:
        k = self.keys[self.current_key_idx]
        return f"...{k[-6:]}" if len(k) > 6 else k

    def _init_chat(self, history=None):
        api_key = self.keys[self.current_key_idx]
        model = self.models[self.current_model_idx]
        self.client = genai.Client(api_key=api_key)

        decls = [
            {
                "name": t.name,
                "description": (t.description or t.name)[:500],
                "parameters": sanitize_schema(t.input_schema),
            }
            for t in self.tools
        ]
        gtool = types.Tool(function_declarations=decls)
        config = types.GenerateContentConfig(
            tools=[gtool],
            system_instruction=SYSTEM_INSTRUCTION,
        )
        self.chat = self.client.chats.create(
            model=model,
            config=config,
            history=history,
        )

    def failover_to_next_key(self) -> bool:
        """Rotates to the next Gemini API key if quota (429) is hit."""
        if self.current_key_idx + 1 < len(self.keys):
            self.current_key_idx += 1
            print(f"\n[!] Gemini Key Quota exhausted. Rotating to Backup Key #{self.current_key_idx + 1} ({self.current_key_preview})...")
            try:
                hist = self.chat.get_history()
            except Exception:
                hist = None
            self._init_chat(history=hist)
            return True
        return False

    def failover_to_next_model(self) -> bool:
        """Switches to the next model in the priority cascade, preserving chat history."""
        old_model = self.current_model
        if self.current_model_idx + 1 < len(self.models):
            self.current_model_idx += 1
            new_model = self.models[self.current_model_idx]
            print(f"\n[!] Model '{old_model}' busy/unavailable. Auto-switching to '{new_model}'...")
            try:
                hist = self.chat.get_history()
            except Exception:
                hist = None
            self._init_chat(history=hist)
            return True
        return False

    def send_message(self, user_input: str, mcp: JiraMCPClient) -> str:
        """Sends message with automated tool calling and failover cascade on 503, 429, or model errors."""
        content_to_send: Any = user_input

        while True:
            try:
                response = self.chat.send_message(content_to_send)

                # Tool calling loop
                while response.function_calls:
                    function_parts = []
                    for call in response.function_calls:
                        tool_name = call.name
                        tool_args = dict(call.args) if call.args else {}

                        print(f"\n[*] [MCP Execution] ({self.current_model}) Calling: {tool_name}")
                        if tool_args:
                            print(f"    Parameters: {tool_args}")

                        try:
                            t_result = mcp.call_tool(tool_name, tool_args)
                            result_payload = t_result.data if t_result.data is not None else t_result.text
                        except MCPError as me:
                            result_payload = {"error": str(me)}

                        tool_summary = format_tool_result_summary(tool_name, result_payload)
                        print(f"    {tool_summary}")

                        if isinstance(result_payload, str) and len(result_payload) > 40000:
                            result_payload = result_payload[:40000] + "\n... [truncated]"

                        function_parts.append(
                            types.Part.from_function_response(
                                name=tool_name,
                                response={"result": result_payload},
                            )
                        )

                    response = self.chat.send_message(function_parts)

                return response.text or "[No response text generated]"

            except Exception as e:
                err_str = str(e).lower()

                # Quota / rate limit (429) -> Try backup key
                if any(x in err_str for x in ["429", "quota", "resource_exhausted"]):
                    if self.failover_to_next_key():
                        content_to_send = user_input
                        continue

                # High demand (503), unavailable, or server error -> Cascade to next model
                if any(x in err_str for x in ["503", "unavailable", "overloaded", "spikes in demand", "404", "500", "502", "504", "429"]):
                    if self.failover_to_next_model():
                        content_to_send = user_input
                        continue

                raise e


# ============================================================================
# Groq Cross-Provider Fallback (Llama 3.3 70B via HTTP)
# ============================================================================

class GroqFallback:
    def __init__(self, api_key: str, model: str, tools: list):
        self.api_key = api_key
        self.model = model
        self.tools = tools
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def send_message(self, user_input: str, mcp: JiraMCPClient) -> str:
        self.history.append({"role": "user", "content": user_input})

        groq_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or t.name)[:1000],
                    "parameters": t.input_schema if t.input_schema else {"type": "object", "properties": {}},
                },
            }
            for t in self.tools
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=60.0) as client:
            while True:
                payload = {
                    "model": self.model,
                    "messages": self.history,
                    "tools": groq_tools,
                    "tool_choice": "auto",
                }

                resp = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                message = choice["message"]

                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    assistant_text = message.get("content") or ""
                    self.history.append({"role": "assistant", "content": assistant_text})
                    return assistant_text

                self.history.append(message)

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"].get("arguments", "{}"))
                    except Exception:
                        fn_args = {}

                    print(f"\n[*] [MCP Execution (Groq)] Calling: {fn_name}")
                    if fn_args:
                        print(f"    Parameters: {fn_args}")

                    try:
                        t_result = mcp.call_tool(fn_name, fn_args)
                        result_text = t_result.text
                        result_payload = t_result.data if t_result.data is not None else t_result.text
                    except Exception as me:
                        result_text = json.dumps({"error": str(me)})
                        result_payload = {"error": str(me)}

                    tool_summary = format_tool_result_summary(fn_name, result_payload)
                    print(f"    {tool_summary}")

                    if len(result_text) > 40000:
                        result_text = result_text[:40000] + "\n... [truncated]"

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": result_text,
                    })


# ============================================================================
# Helpers
# ============================================================================

def load_keys() -> List[str]:
    """Extracts all configured Gemini API keys (primary, backup, or comma-separated list)."""
    keys = []
    primary = os.getenv("GEMINI_API_KEY", "").strip()
    if primary:
        keys.append(primary)

    backup = os.getenv("GEMINI_BACKUP_API_KEY", "").strip()
    if backup and backup not in keys:
        keys.append(backup)

    multi = os.getenv("GEMINI_API_KEYS", "").strip()
    if multi:
        for k in multi.split(","):
            k_clean = k.strip()
            if k_clean and k_clean not in keys:
                keys.append(k_clean)

    return keys


# ============================================================================
# Main Orchestrator REPL
# ============================================================================

def run_orchestrator():
    print("=" * 74)
    print("        [TriageIQ] (Jira MCP Defect Triage & Intelligence)")
    print("=" * 74)

    # 1. Parse Gemini Keys
    gemini_keys = load_keys()
    if not gemini_keys:
        print("[-] Error: No GEMINI_API_KEY found in .env")
        sys.exit(1)

    # 2. Parse Model Cascade
    raw_models = os.getenv(
        "GEMINI_MODELS",
        "gemini-3.5-flash-lite,gemini-flash-lite-latest,gemini-3.1-flash-lite,gemini-3.5-flash"
    )
    models_cascade = [m.strip() for m in raw_models.split(",") if m.strip()]

    # 3. Parse Groq Config
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    print(f"[+] Active Gemini Keys: {len(gemini_keys)} key(s) loaded")
    print(f"[+] Fallback Model Cascade: {' -> '.join(models_cascade)}")
    if groq_key:
        print(f"[+] Groq Cross-Provider Fallback: ENABLED ({groq_model})")
    else:
        print(f"[*] Groq Fallback Slot: READY in .env (add GROQ_API_KEY to enable)")

    target_jira = os.getenv("JIRA_URL") or os.getenv("JIRA_BASE_URL", "https://your-domain.atlassian.net")
    target_project = os.getenv("JIRA_PROJECT_KEY", "SCRUM")
    print(f"[+] Target Jira Instance: {target_jira} (Default Project: {target_project})")
    print(f"[+] Vector Database: ChromaDB (jira_defects index)")

    print("\n[+] Starting Smart Jira MCP Server & discovering tools...")

    with JiraMCPClient() as mcp:
        tools = mcp.list_tools()
        print(f"[+] Connected! Discovered {len(tools)} tools from Smart Jira MCP Server.")

        gemini_agent = GeminiOrchestrator(keys=gemini_keys, models=models_cascade, tools=tools)
        groq_agent = GroqFallback(api_key=groq_key, model=groq_model, tools=tools)

        print(f"\n[+] Active Engine: {gemini_agent.current_model} (Primary)")
        read_only_mode = os.getenv("JIRA_READ_ONLY", "true").lower() in ("true", "1", "yes")
        print("\nAsk anything in natural language! Examples:")
        print("  - 'Node.js process crashes out of memory every 4 hours under peak traffic when rendering invoices'")
        print("  - 'Triage defect SCRUM-1 and recommend root cause investigation steps'")
        print("  - 'List open issues in project SCRUM'")
        print("  - 'Inspect details and recent history for SCRUM-1'")
        if not read_only_mode:
            print("  - 'Create a high priority bug for checkout failure'")
            print("  - 'Move SCRUM-2 to Done'")
        else:
            print("  - (Note: Ticket mutation tools require JIRA_READ_ONLY=false)")
        print("  - Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                user_input = input("\nYou > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[+] Exiting...")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("[+] Exiting Jira AI Orchestrator. Goodbye!")
                break

            print("\n[Thinking...]")

            try:
                answer = gemini_agent.send_message(user_input, mcp)
                print(f"\n[AI Assistant ({gemini_agent.current_model})]:\n{answer}\n")
                print("-" * 74)

            except Exception as e:
                if groq_agent.is_configured():
                    print(f"\n[!] Gemini cascade exhausted ({e}). Failing over to Groq ({groq_model})...")
                    try:
                        answer = groq_agent.send_message(user_input, mcp)
                        print(f"\n[AI Assistant (Groq - {groq_model})]:\n{answer}\n")
                        print("-" * 74)
                    except Exception as ge:
                        print(f"\n[-] Both Gemini and Groq failover failed: {ge}\n")
                else:
                    print(f"\n[-] Error across all Gemini fallback models: {e}")
                    print("[!] Tip: You can paste a Groq API key in .env (GROQ_API_KEY) or a secondary Gemini key to enable backup.\n")


if __name__ == "__main__":
    run_orchestrator()
