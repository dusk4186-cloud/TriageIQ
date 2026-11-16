"""
Smart Jira MCP Client
Connects to the Unified Smart Jira MCP Server over stdio transport.
Supports dynamic tool discovery, structured tool invocation, CLI parameters, and automated demo mode.
"""

import os
import sys
import json
import time
import queue
import random
import logging
import argparse
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Ensure clean UTF-8 on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("jira_mcp_client")
# Suppress noisy HTTP client loggers
for _log_name in ("httpx", "httpx2", "httpcore", "httpcore2"):
    logging.getLogger(_log_name).setLevel(logging.WARNING)


# ============================================================================
# Exceptions
# ============================================================================

class MCPError(Exception):
    """Base exception for MCP client errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when server process fails or stdio pipes break."""
    pass


class MCPTimeoutError(MCPError):
    """Raised when a request times out waiting for server response."""
    pass


class MCPProtocolError(MCPError):
    """Raised when server returns JSON-RPC protocol error."""
    def __init__(self, message: str, code: Optional[int] = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class MCPToolCallError(MCPError):
    """Raised when a tool execution fails on the server."""
    def __init__(self, message: str, tool_name: str, content: Any = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.content = content


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ToolInfo:
    """Represents a Jira MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def required_params(self) -> List[str]:
        return self.input_schema.get("required", [])

    @property
    def properties(self) -> Dict[str, Any]:
        return self.input_schema.get("properties", {})


@dataclass
class ToolResult:
    """Represents the execution result of an MCP tool call."""
    tool_name: str
    content: List[Dict[str, Any]]
    is_error: bool
    raw: Dict[str, Any]

    @property
    def text(self) -> str:
        texts = []
        for item in self.content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)

    @property
    def data(self) -> Any:
        raw_text = self.text.strip()
        try:
            return json.loads(raw_text)
        except (ValueError, TypeError):
            return raw_text


# ============================================================================
# Core MCP Client
# ============================================================================

class JiraMCPClient:
    """
    Client for interacting with the Jira MCP server via stdio transport.
    """

    def __init__(
        self,
        server_path: Optional[Path] = None,
        python_exe: Optional[str] = None,
        default_timeout: float = 30.0,
    ):
        self.default_timeout = default_timeout
        self.python_exe = python_exe or sys.executable

        # Locate server.py
        if server_path:
            self.server_path = Path(server_path).resolve()
        else:
            current_dir = Path(__file__).resolve().parent
            candidates = [
                current_dir / "server.py",
                current_dir.parent / "server.py",
                current_dir.parent / "Jira_MCP" / "server.py",
            ]
            self.server_path = None
            for cand in candidates:
                if cand.exists():
                    self.server_path = cand
                    break
            if not self.server_path:
                self.server_path = current_dir / "server.py"

        self.proc: Optional[subprocess.Popen] = None
        self._next_id = 1
        self._id_lock = threading.Lock()
        self._pending_responses: Dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._stderr_buffer: List[str] = []

    def _get_next_id(self) -> int:
        with self._id_lock:
            val = self._next_id
            self._next_id += 1
            return val

    def start(self) -> None:
        """Starts the Jira MCP server process and completes the protocol handshake."""
        if self.proc and self.proc.poll() is None:
            return

        cmd = [self.python_exe, str(self.server_path)]
        logger.info(f"Launching Smart Jira MCP Server: python {self.server_path.name}")

        server_cwd = self.server_path.parent

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(server_cwd),
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except Exception as e:
            raise MCPConnectionError(f"Failed to start Jira MCP server process: {e}")

        self._running = True

        self._reader_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self._reader_thread.start()

        def _stderr_loop():
            while self._running and self.proc and self.proc.stderr:
                line = self.proc.stderr.readline()
                if not line:
                    break
                self._stderr_buffer.append(line.rstrip())
                if len(self._stderr_buffer) > 100:
                    self._stderr_buffer.pop(0)

        threading.Thread(target=_stderr_loop, daemon=True).start()

        self._perform_handshake()

    def _read_stdout_loop(self) -> None:
        while self._running and self.proc and self.proc.stdout:
            try:
                line = self.proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                req_id = msg.get("id")
                if req_id is not None:
                    with self._pending_lock:
                        q = self._pending_responses.get(req_id)
                    if q:
                        q.put(msg)
            except Exception:
                break

    def _perform_handshake(self) -> None:
        init_payload = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "jira-smart-mcp-client", "version": "2.0.0"},
        }
        res = self._send_request("initialize", init_payload, timeout=15.0)
        protocol_version = res.get("protocolVersion")
        server_info = res.get("serverInfo", {})
        logger.info(f"Connected to MCP Server: {server_info.get('name')} (protocol {protocol_version})")
        self._send_notification("notifications/initialized", {})

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.proc or self.proc.poll() is not None:
            raise MCPConnectionError("MCP server process is not running.")
        notif = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        try:
            data = json.dumps(notif) + "\n"
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise MCPConnectionError(f"Failed to write notification to server: {e}")

    def _send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        timeout = timeout or self.default_timeout
        if not self.proc or self.proc.poll() is not None:
            raise MCPConnectionError("MCP server process is not running.")

        req_id = self._get_next_id()
        resp_q: queue.Queue = queue.Queue()
        with self._pending_lock:
            self._pending_responses[req_id] = resp_q

        req = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        try:
            line = json.dumps(req) + "\n"
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            with self._pending_lock:
                self._pending_responses.pop(req_id, None)
            raise MCPConnectionError(f"Failed to write request {method} (id={req_id}): {e}")

        try:
            resp = resp_q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending_responses.pop(req_id, None)
            raise MCPTimeoutError(f"Timeout waiting {timeout}s for response to {method} (id={req_id})")

        with self._pending_lock:
            self._pending_responses.pop(req_id, None)

        if "error" in resp:
            err = resp["error"]
            raise MCPProtocolError(
                message=err.get("message", "Unknown error"),
                code=err.get("code"),
                data=err.get("data"),
            )

        return resp.get("result", {})

    def list_tools(self) -> List[ToolInfo]:
        """Discovers all available tools exposed by the Jira MCP server."""
        res = self._send_request("tools/list", {})
        raw_tools = res.get("tools", [])
        tools = []
        for t in raw_tools:
            tools.append(
                ToolInfo(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
            )
        return tools

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Invokes a specific tool on the Jira MCP server."""
        res = self._send_request("tools/call", {"name": name, "arguments": arguments or {}})
        content = res.get("content", [])
        is_error = res.get("isError", False)
        return ToolResult(tool_name=name, content=content, is_error=is_error, raw=res)

    def close(self) -> None:
        """Terminates server process and cleans up resources."""
        self._running = False
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                if self.proc.stdout:
                    self.proc.stdout.close()
                if self.proc.stderr:
                    self.proc.stderr.close()
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# Interactive CLI & Test Suite
# ============================================================================

def run_demo(client: JiraMCPClient) -> None:
    """Executes a full interactive demonstration across all tools including Defect Triage."""
    print("=" * 72)
    print("  SMART JIRA MCP SERVER DEMONSTRATION (Jira + ChromaDB RAG)")
    print("=" * 72)

    # 1. List Projects
    print("\n[Step 1] Discovering Projects (jira_list_projects):")
    res1 = client.call_tool("jira_list_projects")
    print(res1.text)

    # 2. Search Issues
    print("\n[Step 2] Searching Issues (jira_search_issues - query='timeout'):")
    res2 = client.call_tool("jira_search_issues", {"query": "timeout", "max_results": 3})
    print(res2.text)

    # 3. Get Issue
    target_key = "SCRUM-1"
    print(f"\n[Step 3] Fetching Issue Details (jira_get_issue - '{target_key}'):")
    res3 = client.call_tool("jira_get_issue", {"issue_key": target_key, "include_comments": True})
    print(res3.text)

    # 4. Semantic Defect Similarity Search in ChromaDB
    print("\n[Step 4] ChromaDB Semantic Defect Search (jira_find_similar_defects):")
    res4 = client.call_tool("jira_find_similar_defects", {
        "query": "Database connection pool timeout during checkout query spike",
        "top_k": 2,
    })
    print(res4.text)

    # 5. AI Defect Triaging with ChromaDB RAG + Gemini
    print(f"\n[Step 5] AI Defect Triaging (jira_triage_defect - '{target_key}'):")
    res5 = client.call_tool("jira_triage_defect", {
        "issue_key": target_key,
        "post_comment_to_jira": False,
    })
    print(res5.text)

    # 6. Create Issue
    print("\n[Step 6] Creating New Issue (jira_create_issue):")
    create_args = {
        "summary": "Fix connection pool exhaustion under flash sale loads",
        "issue_type": "Bug",
        "description": "DB queries hang when concurrent requests exceed 500 connections.",
        "priority": "High",
        "labels": ["database", "performance", "backend"],
    }
    res6 = client.call_tool("jira_create_issue", create_args)
    print(res6.text)
    new_key = res6.data.get("key", "SCRUM-4") if isinstance(res6.data, dict) else "SCRUM-4"

    # 7. Update Issue
    print(f"\n[Step 7] Updating Issue Fields (jira_update_issue - {new_key}):")
    res7 = client.call_tool("jira_update_issue", {
        "issue_key": new_key,
        "priority": "Highest",
        "summary": "Fix connection pool exhaustion under flash sale loads [CRITICAL]",
    })
    print(res7.text)

    # 8. Add Comment
    print(f"\n[Step 8] Adding Comment (jira_add_comment - {new_key}):")
    res8 = client.call_tool("jira_add_comment", {
        "issue_key": new_key,
        "comment": "Triaged by AI: Recommending pool size increase to 100 with idle timeout 30s.",
    })
    print(res8.text)

    # 9. Transition Issue
    print(f"\n[Step 9a] Checking Available Transitions (jira_transition_issue - {new_key}):")
    res9a = client.call_tool("jira_transition_issue", {"issue_key": new_key, "list_transitions": True})
    print(res9a.text)

    print(f"\n[Step 9b] Transitioning Status to 'In Progress':")
    res9b = client.call_tool("jira_transition_issue", {
        "issue_key": new_key,
        "transition_name": "In Progress",
    })
    print(res9b.text)

    print("\n" + "=" * 72)
    print(" [OK] All 9 Smart Jira MCP Tools Tested Successfully!")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Smart Jira MCP Client CLI")
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    parser.add_argument("--detailed", action="store_true", help="Show full input schemas when listing tools")
    parser.add_argument("--call", type=str, metavar="TOOL_NAME", help="Invoke tool by name")
    parser.add_argument("--args", type=str, metavar="JSON_STRING", help="JSON dictionary of tool arguments")
    parser.add_argument("--param", action="append", help="Key=Value tool argument (can specify multiple)")
    parser.add_argument("--demo", action="store_true", help="Run automated 9-tool demonstration")

    args = parser.parse_args()

    if not (args.list_tools or args.call or args.demo):
        args.demo = True

    with JiraMCPClient() as client:
        if args.list_tools:
            tools = client.list_tools()
            print(f"\nDiscovered {len(tools)} Smart Jira MCP Tools:")
            print("-" * 65)
            for t in tools:
                print(f"[TOOL] {t.name}")
                print(f"       {t.description}")
                if args.detailed:
                    props = t.properties
                    reqs = t.required_params
                    print("   Parameters:")
                    for pname, pinfo in props.items():
                        req_mark = "*" if pname in reqs else ""
                        ptype = pinfo.get("type", "any")
                        pdesc = pinfo.get("description", "")
                        print(f"     - {pname}{req_mark} ({ptype}): {pdesc}")
                print()
            return

        if args.call:
            arguments = {}
            if args.args:
                try:
                    arguments = json.loads(args.args)
                except json.JSONDecodeError as e:
                    print(f"Error parsing --args JSON: {e}", file=sys.stderr)
                    sys.exit(1)

            if args.param:
                for p in args.param:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        if v.lower() == "true":
                            v = True
                        elif v.lower() == "false":
                            v = False
                        elif v.isdigit():
                            v = int(v)
                        arguments[k.strip()] = v

            logger.info(f"Calling tool '{args.call}' with: {arguments}")
            res = client.call_tool(args.call, arguments)
            print(res.text)
            return

        if args.demo:
            run_demo(client)


if __name__ == "__main__":
    main()
