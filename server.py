"""
Unified Smart Jira MCP Server
Provides standard Jira issue lifecycle tools + AI Defect Triaging with ChromaDB RAG & Gemini:
1. jira_list_projects - Discover accessible projects
2. jira_search_issues - Query issues using JQL or text search
3. jira_get_issue - Fetch full issue details, description, and comments
4. jira_create_issue - Create new tasks, bugs, and stories (default: SCRUM project)
5. jira_update_issue - Update issue fields
6. jira_add_comment - Post comments and status updates
7. jira_transition_issue - Move issues through workflow states or list valid transitions
8. jira_find_similar_defects - Semantic vector search over historical defects in ChromaDB
9. jira_triage_defect - End-to-end AI defect root-cause analysis & auto-commenting via Gemini
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
env_paths = [
    BASE_DIR / ".env",
    BASE_DIR / "Defect Triage" / ".env",
    Path.cwd() / ".env",
]
for p in env_paths:
    if p.exists():
        load_dotenv(p)
        break

from mcp.server.mcpserver import MCPServer
from jira_client import JiraClient, JiraAPIError

# Logging to stderr keeps stdout clean for MCP JSON-RPC protocol
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("jira_mcp_server")

# Initialize MCPServer instance
app = MCPServer("jira-smart-mcp-server")

_client: Optional[JiraClient] = None


def get_jira_client() -> JiraClient:
    """Returns or lazily instantiates the Jira REST client."""
    global _client
    if _client is None:
        _client = JiraClient()
    return _client


def _format_response(data: Any) -> str:
    """Helper to return pretty-printed JSON string for MCP text content."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def check_read_only(tool_name: str) -> Optional[str]:
    """
    Enforces read-only safety.
    When JIRA_READ_ONLY=true (default: true), mutating operations (create, update,
    comment, transition) are blocked to prevent unauthorized writes or prompt injection.
    """
    if os.getenv("JIRA_READ_ONLY", "true").lower() in ("true", "1", "yes"):
        logger.warning(f"Blocked write attempt on '{tool_name}' because JIRA_READ_ONLY=true")
        return _format_response({
            "error": (
                f"Write operation '{tool_name}' blocked: Server is running in read-only mode (JIRA_READ_ONLY=true). "
                "Set JIRA_READ_ONLY=false in .env to permit modifying tickets."
            ),
            "status": "blocked",
            "read_only": True,
        })
    return None


# ============================================================================
# Tool 1: List Projects
# ============================================================================

@app.tool(
    name="jira_list_projects",
    description="List accessible Jira projects with their key, name, ID, and project type.",
)
def jira_list_projects() -> str:
    """List accessible Jira projects."""
    try:
        client = get_jira_client()
        projects = client.list_projects()
        return _format_response({"count": len(projects), "projects": projects})
    except JiraAPIError as e:
        return _format_response({"error": str(e), "status_code": e.status_code})
    except Exception as e:
        logger.exception("Unexpected error in jira_list_projects")
        return _format_response({"error": f"Internal server error: {e}"})


# ============================================================================
# Tool 2: Search Issues
# ============================================================================

@app.tool(
    name="jira_search_issues",
    description=(
        "Search Jira issues using JQL (e.g. `project = SCRUM AND status = 'In Progress'`) "
        "or free-text search keywords. Returns summary, status, priority, assignee, and issue URL."
    ),
)
def jira_search_issues(
    query: str,
    max_results: int = 10,
) -> str:
    """
    Search issues in Jira.
    :param query: JQL query string or free-text keywords to search for.
    :param max_results: Maximum number of issues to return (default: 10, max: 50).
    """
    try:
        client = get_jira_client()
        issues = client.search_issues(query=query, max_results=max_results)
        return _format_response({
            "query": query,
            "count": len(issues),
            "issues": issues,
        })
    except JiraAPIError as e:
        return _format_response({"error": str(e), "status_code": e.status_code})
    except Exception as e:
        logger.exception("Unexpected error in jira_search_issues")
        return _format_response({"error": f"Internal server error: {e}"})


# ============================================================================
# Tool 3: Get Issue
# ============================================================================

@app.tool(
    name="jira_get_issue",
    description=(
        "Retrieve full details of a Jira issue by key (e.g. 'SCRUM-1'), including summary, "
        "clean text description, current status, priority, assignee, reporter, labels, and recent comments."
    ),
)
def jira_get_issue(
    issue_key: str,
    include_comments: bool = True,
) -> str:
    """
    Get full details of a specific Jira issue.
    :param issue_key: The issue key (e.g. 'SCRUM-1').
    :param include_comments: Whether to include recent comments (default: True).
    """
    try:
        client = get_jira_client()
        issue = client.get_issue(issue_key=issue_key, include_comments=include_comments)
        return _format_response(issue)
    except JiraAPIError as e:
        return _format_response({"error": str(e), "status_code": e.status_code})
    except Exception as e:
        logger.exception(f"Unexpected error in jira_get_issue for {issue_key}")
        return _format_response({"error": f"Internal server error: {e}"})


# ============================================================================
# Tool 4: Create Issue
# ============================================================================

@app.tool(
    name="jira_create_issue",
    description=(
        "Create a new Jira issue (Task, Bug, Story) with a summary, description, "
        "priority, and optional assignee and labels. Defaults to configured project (SCRUM)."
    ),
)
def jira_create_issue(
    summary: str,
    project_key: Optional[str] = None,
    issue_type: str = "Task",
    description: str = "",
    priority: str = "Medium",
    assignee: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> str:
    """
    Create a new issue in a Jira project.
    :param summary: Brief title or summary of the issue.
    :param project_key: Project identifier (optional, defaults to JIRA_PROJECT_KEY e.g. 'SCRUM').
    :param issue_type: Issue type name (default: 'Task', e.g. 'Bug', 'Story').
    :param description: Detailed markdown/plain text description of the issue.
    :param priority: Issue priority ('Lowest', 'Low', 'Medium', 'High', 'Highest').
    :param assignee: Optional username, email, or accountId of the assignee.
    :param labels: Optional list of label strings.
    """
    blocked = check_read_only("jira_create_issue")
    if blocked:
        return blocked

    try:
        client = get_jira_client()
        result = client.create_issue(
            summary=summary,
            project_key=project_key,
            issue_type=issue_type,
            description=description,
            priority=priority,
            assignee=assignee,
            labels=labels,
        )
        return _format_response(result)
    except JiraAPIError as e:
        return _format_response({"error": str(e), "status_code": e.status_code})
    except Exception as e:
        logger.exception("Unexpected error in jira_create_issue")
        return _format_response({"error": f"Internal server error: {e}"})


# ============================================================================
# Tool 5: Update Issue
# ============================================================================

@app.tool(
    name="jira_update_issue",
    description=(
        "Update fields of an existing Jira issue by key (e.g. 'SCRUM-1'). "
        "Specify any combination of summary, description, priority, assignee, or labels to update."
    ),
)
def jira_update_issue(
    issue_key: str,
    summary: Optional[str] = None,
    description: Optional[str] = None,
    priority: Optional[str] = None,
    assignee: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> str:
    """
    Update fields of an existing issue.
    :param issue_key: The issue key (e.g. 'SCRUM-1').
    :param summary: New summary/title (optional).
    :param description: New markdown/plain text description (optional).
    :param priority: New priority (optional).
    :param assignee: New assignee username/accountId (optional).
    :param labels: New list of labels (optional).
    """
    blocked = check_read_only("jira_update_issue")
    if blocked:
        return blocked

    try:
        client = get_jira_client()
        result = client.update_issue(
            issue_key=issue_key,
            summary=summary,
            description=description,
            priority=priority,
            assignee=assignee,
            labels=labels,
        )
        return _format_response(result)
    except JiraAPIError as e:
        return _format_response({"error": str(e), "status_code": e.status_code})
    except Exception as e:
        logger.exception(f"Unexpected error in jira_update_issue for {issue_key}")
        return _format_response({"error": f"Internal server error: {e}"})


# ============================================================================
# Tool 6: Add Comment
# ============================================================================

@app.tool(
    name="jira_add_comment",
    description="Add a comment to an existing Jira issue by key (e.g. 'SCRUM-1').",
)
def jira_add_comment(
    issue_key: str,
    comment: str,
) -> str:
    """
    Add a comment to a Jira issue.
    :param issue_key: The issue key (e.g. 'SCRUM-1').
    :param comment: Plain text or markdown comment content.
    """
    blocked = check_read_only("jira_add_comment")
    if blocked:
        return blocked

    try:
        client = get_jira_client()
        result = client.add_comment(issue_key=issue_key, comment=comment)
        return _format_response(result)
    except JiraAPIError as e:
        return _format_response({"error": str(e), "status_code": e.status_code})
    except Exception as e:
        logger.exception(f"Unexpected error in jira_add_comment for {issue_key}")
        return _format_response({"error": f"Internal server error: {e}"})


# ============================================================================
# Tool 7: Transition Issue
# ============================================================================

@app.tool(
    name="jira_transition_issue",
    description=(
        "Change the workflow state/status of a Jira issue (e.g. move to 'In Progress', 'Done', 'To Do'). "
        "Set list_transitions=True or omit transition_name to see all available transitions for the issue."
    ),
)
def jira_transition_issue(
    issue_key: str,
    transition_name: Optional[str] = None,
    list_transitions: bool = False,
) -> str:
    """
    Transition an issue to a new status, or list available transitions.
    :param issue_key: The issue key (e.g. 'SCRUM-1').
    :param transition_name: Name of target status or transition (e.g. 'Done', 'In Progress').
    :param list_transitions: If True, returns available next statuses without moving the issue.
    """
    if transition_name:
        blocked = check_read_only("jira_transition_issue")
        if blocked:
            return blocked

    try:
        client = get_jira_client()
        result = client.transition_issue(
            issue_key=issue_key,
            transition_name=transition_name,
            list_transitions=list_transitions,
        )
        return _format_response(result)
    except JiraAPIError as e:
        return _format_response({"error": str(e), "status_code": e.status_code})
    except Exception as e:
        logger.exception(f"Unexpected error in jira_transition_issue for {issue_key}")
        return _format_response({"error": f"Internal server error: {e}"})


# ============================================================================
# Tool 8: Find Similar Historical Defects (ChromaDB Vector RAG)
# ============================================================================

@app.tool(
    name="jira_find_similar_defects",
    description=(
        "Search historical Jira defects in ChromaDB using semantic vector similarity embeddings. "
        "Returns top matching historical defects with similarity scores, descriptions, and known root causes."
    ),
)
def jira_find_similar_defects(
    query: str,
    top_k: int = 3,
) -> str:
    """
    Semantic vector search over historical Jira defects.
    :param query: Bug description, title, or error log to compare against historical defects.
    :param top_k: Number of most similar historical defects to retrieve (default: 3).
    """
    try:
        from defect_triage_engine import search_similar_defects
        matches = search_similar_defects(query_text=query, top_k=top_k)
        return _format_response({
            "query": query,
            "match_count": len(matches),
            "matches": matches,
        })
    except Exception as e:
        logger.exception("Error in jira_find_similar_defects")
        return _format_response({"error": f"Vector similarity search failed: {e}"})


# ============================================================================
# Tool 9: AI Defect Triaging (ChromaDB + Gemini AI)
# ============================================================================

@app.tool(
    name="jira_triage_defect",
    description=(
        "Perform AI-powered defect triaging on a Jira issue. Fetches the defect from Jira, "
        "retrieves similar historical bugs from ChromaDB, analyzes root causes with Gemini, "
        "and optionally posts the triage recommendation directly back to the Jira ticket as a comment."
    ),
)
def jira_triage_defect(
    issue_key: str,
    post_comment_to_jira: bool = False,
) -> str:
    """
    Triages a Jira defect using ChromaDB historical evidence + Gemini AI.
    :param issue_key: The Jira issue key to triage (e.g. 'SCRUM-1').
    :param post_comment_to_jira: If True, automatically adds the generated triage recommendation as a comment on the ticket.
    """
    try:
        client = get_jira_client()
        issue = client.get_issue(issue_key=issue_key, include_comments=False)
        summary = issue.get("summary", "")
        description = issue.get("description", "")

        from defect_triage_engine import search_similar_defects, run_defect_triage

        # Search similar historical defects
        search_query = f"Title: {summary}\n\nDescription: {description}"
        matches = search_similar_defects(query_text=search_query, top_k=3)

        # Generate Gemini triage recommendation
        recommendation = run_defect_triage(
            issue_key=issue_key,
            summary=summary,
            description=description,
            matches=matches,
        )

        comment_posted = False
        comment_id = None
        if post_comment_to_jira:
            blocked = check_read_only("jira_triage_defect (post_comment_to_jira)")
            if blocked:
                return blocked
            comment_body = (
                f"🤖 **Automated Defect Triage Analysis (Gemini + ChromaDB)**\n\n"
                f"{recommendation}"
            )
            comment_res = client.add_comment(issue_key=issue_key, comment=comment_body)
            comment_posted = True
            comment_id = comment_res.get("id")

        return _format_response({
            "issue_key": issue_key,
            "summary": summary,
            "similar_defects_count": len(matches),
            "similar_defects": matches,
            "recommendation": recommendation,
            "comment_posted": comment_posted,
            "comment_id": comment_id,
        })

    except JiraAPIError as e:
        return _format_response({"error": str(e), "status_code": e.status_code})
    except Exception as e:
        logger.exception(f"Error in jira_triage_defect for {issue_key}")
        return _format_response({"error": f"Triage failed: {e}"})


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Smart Jira MCP Server (Standard Tools + Defect Triage RAG)...")
    app.run(transport="stdio")
