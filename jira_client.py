"""
Jira REST API Client
Supports Atlassian Cloud (Email + API Token) and Jira Server/Data Center (Bearer/PAT).
Includes ADF (Atlassian Document Format) parsing, error formatting, and mock mode fallback.
"""

import os
import re
import json
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

logger = logging.getLogger("jira_client")


def adf_to_text(node: Any) -> str:
    """Recursively converts Atlassian Document Format (ADF) node to clean markdown/plain text."""
    if not node:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(item) for item in node)
    if not isinstance(node, dict):
        return str(node)

    node_type = node.get("type", "")
    content = node.get("content", [])

    if node_type == "text":
        return node.get("text", "")
    elif node_type == "hardBreak":
        return "\n"
    elif node_type == "paragraph":
        inner = "".join(adf_to_text(child) for child in content)
        return inner + "\n"
    elif node_type == "heading":
        level = node.get("attrs", {}).get("level", 1)
        inner = "".join(adf_to_text(child) for child in content)
        return f"{'#' * level} {inner}\n"
    elif node_type in ("bulletList", "orderedList"):
        return "".join(adf_to_text(child) for child in content)
    elif node_type == "listItem":
        inner = "".join(adf_to_text(child) for child in content).strip()
        return f"- {inner}\n"
    elif node_type == "codeBlock":
        lang = node.get("attrs", {}).get("language", "")
        inner = "".join(adf_to_text(child) for child in content)
        return f"```{lang}\n{inner}\n```\n"
    elif node_type == "blockquote":
        inner = "".join(adf_to_text(child) for child in content)
        return f"> {inner}\n"
    else:
        return "".join(adf_to_text(child) for child in content)


def text_to_adf(text: str) -> Dict[str, Any]:
    """Converts plain text/markdown string to minimal valid Atlassian Document Format (ADF)."""
    if not text:
        return {"version": 1, "type": "doc", "content": []}

    paragraphs = text.split("\n\n")
    content = []
    for para in paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue
        lines = cleaned.split("\n")
        para_content = []
        for i, line in enumerate(lines):
            if i > 0:
                para_content.append({"type": "hardBreak"})
            para_content.append({"type": "text", "text": line})
        content.append({"type": "paragraph", "content": para_content})

    if not content:
        content = [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]

    return {"version": 1, "type": "doc", "content": content}


class JiraAPIError(Exception):
    """Custom exception with user-friendly Jira error explanation."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class JiraClient:
    """
    High-level Jira REST client handling authentication, request formatting,
    response parsing, and offline mock fallback.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        bearer_token: Optional[str] = None,
        project_key: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        read_only: Optional[bool] = None,
        timeout: float = 15.0,
    ):
        raw_url = base_url or os.getenv("JIRA_URL") or os.getenv("JIRA_BASE_URL", "")
        self.base_url = raw_url.rstrip("/")
        self.email = email or os.getenv("JIRA_EMAIL", "")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN", "")
        self.bearer_token = bearer_token or os.getenv("JIRA_BEARER_TOKEN", "") or os.getenv("JIRA_PAT", "")
        self.default_project_key = (project_key or os.getenv("JIRA_PROJECT_KEY", "SCRUM")).strip().upper()
        self.timeout = timeout

        # Read-only default: Defaults to True if not explicitly disabled
        env_read_only = os.getenv("JIRA_READ_ONLY", "true").lower() in ("true", "1", "yes")
        self.read_only = read_only if read_only is not None else env_read_only

        env_mock = os.getenv("JIRA_MOCK_MODE", "").lower() in ("true", "1", "yes")
        if mock_mode is not None:
            self.mock_mode = mock_mode
        elif env_mock:
            self.mock_mode = True
        elif not self.base_url or (not self.api_token and not self.bearer_token):
            self.mock_mode = True
            logger.info("Jira credentials not detected; activating Mock Mode for testing.")
        else:
            self.mock_mode = False

        self._session = requests.Session()
        self._setup_auth()

        if self.mock_mode:
            self._init_mock_db()

    def _setup_auth(self) -> None:
        """Sets up HTTP headers based on available credentials."""
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Jira-MCP-Server/2.0",
        })

        if self.bearer_token:
            self._session.headers["Authorization"] = f"Bearer {self.bearer_token}"
        elif self.email and self.api_token:
            auth_str = f"{self.email}:{self.api_token}"
            b64_auth = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
            self._session.headers["Authorization"] = f"Basic {b64_auth}"

    def _check_read_only(self, operation: str) -> None:
        """Enforces read-only safety, raising an error if a mutation is attempted."""
        if self.read_only:
            raise JiraAPIError(
                f"Write operation '{operation}' blocked: JiraClient is operating in read-only mode (JIRA_READ_ONLY=true). "
                "Set JIRA_READ_ONLY=false in your .env file or client constructor to permit write operations.",
                status_code=403,
            )

    def _init_mock_db(self) -> None:
        """Initializes simulated Jira projects, issues, comments, and transitions for offline testing."""
        self._mock_projects = [
            {"key": self.default_project_key or "SCRUM", "name": "Scrum Team Project", "id": "10001", "project_type": "software"},
            {"key": "OPS", "name": "Cloud Operations", "id": "10002", "project_type": "service_desk"},
        ]
        prefix = self.default_project_key or "SCRUM"
        self._mock_issues = {
            f"{prefix}-1": {
                "key": f"{prefix}-1",
                "id": "20001",
                "summary": "Fix authentication token expiration on mobile client",
                "description": "Tokens expire prematurely after 15 minutes instead of the configured 24 hours.",
                "status": "In Progress",
                "priority": "High",
                "issue_type": "Bug",
                "assignee": "Alex Developer",
                "reporter": "Sarah Product",
                "created": "2026-10-10T10:00:00.000+0000",
                "updated": "2026-10-16T14:30:00.000+0000",
                "labels": ["security", "auth", "mobile"],
                "comments": [
                    {
                        "id": "30001",
                        "author": "Sarah Product",
                        "created": "2026-10-10T11:00:00.000+0000",
                        "body": "Reported by multiple beta testers today.",
                    }
                ],
            },
            f"{prefix}-2": {
                "key": f"{prefix}-2",
                "id": "20002",
                "summary": "Checkout payment gateway timeout during high load",
                "description": "Stripe webhook handling drops connections during peak flash sale traffic.",
                "status": "To Do",
                "priority": "Highest",
                "issue_type": "Bug",
                "assignee": "Devin Engineer",
                "reporter": "Alex Developer",
                "created": "2026-10-11T09:15:00.000+0000",
                "updated": "2026-10-11T09:15:00.000+0000",
                "labels": ["backend", "payments", "critical"],
                "comments": [],
            },
            f"{prefix}-3": {
                "key": f"{prefix}-3",
                "id": "20003",
                "summary": "Set up CI/CD pipeline for Jira MCP server deployment",
                "description": "Create automated tests, Dockerfile, and GitHub Actions workflow for stdio and HTTP runners.",
                "status": "Done",
                "priority": "Medium",
                "issue_type": "Story",
                "assignee": "Alex Developer",
                "reporter": "Alex Developer",
                "created": "2026-10-12T14:00:00.000+0000",
                "updated": "2026-10-15T18:00:00.000+0000",
                "labels": ["devops", "mcp"],
                "comments": [],
            },
        }
        self._mock_issue_counter = 4

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Sends an HTTP request to Jira REST API with robust error decoding."""
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.request(
                method=method,
                url=url,
                params=params,
                json=data if data is not None else None,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise JiraAPIError(f"Request to Jira timed out after {self.timeout}s ({url})", status_code=504)
        except requests.exceptions.ConnectionError as e:
            raise JiraAPIError(f"Failed to connect to Jira at {self.base_url}: {e}", status_code=502)

        if resp.status_code >= 400:
            error_msg = f"Jira API error ({resp.status_code})"
            details = None
            try:
                err_data = resp.json()
                details = err_data
                messages = []
                if "errorMessages" in err_data and err_data["errorMessages"]:
                    messages.extend(err_data["errorMessages"])
                if "errors" in err_data and isinstance(err_data["errors"], dict):
                    for field, msg in err_data["errors"].items():
                        messages.append(f"{field}: {msg}")
                if messages:
                    error_msg += f": {'; '.join(messages)}"
                elif "message" in err_data:
                    error_msg += f": {err_data['message']}"
            except Exception:
                error_msg += f": {resp.text.strip()[:200]}"

            if resp.status_code == 401:
                error_msg += ". Please verify JIRA_EMAIL and JIRA_API_TOKEN in .env."
            elif resp.status_code == 403:
                error_msg += ". Access forbidden. Please check your account permissions."
            elif resp.status_code == 404:
                error_msg += ". Resource or issue not found."

            raise JiraAPIError(error_msg, status_code=resp.status_code, details=details)

        if resp.status_code == 204:
            return {}

        try:
            return resp.json()
        except ValueError:
            return resp.text

    # =========================================================================
    # 1. List Projects
    # =========================================================================

    def list_projects(self) -> List[Dict[str, Any]]:
        """Lists accessible Jira projects with key, name, and project type."""
        if self.mock_mode:
            return self._mock_projects

        raw = self._request("GET", "/rest/api/2/project")
        projects = []
        for p in raw:
            projects.append({
                "key": p.get("key"),
                "name": p.get("name"),
                "id": p.get("id"),
                "project_type": p.get("projectTypeKey", "software"),
            })
        return projects

    # =========================================================================
    # 2. Search Issues
    # =========================================================================

    def search_issues(
        self,
        query: str,
        max_results: int = 10,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches issues using JQL or simple keyword search.
        Auto-wraps non-JQL query terms into text search if needed.
        """
        query = (query or "").strip()
        if not query:
            if self.default_project_key:
                jql = f"project = {self.default_project_key} ORDER BY updated DESC"
            else:
                jql = "ORDER BY updated DESC"
        elif any(token in query.lower() for token in ("=", "!=", "~", "order by", "and ", "or ", "is ", "in (")):
            jql = query
        else:
            if re.match(r"^[A-Za-z]+-\d+$", query):
                jql = f"issueKey = {query.upper()}"
            elif self.default_project_key:
                jql = f'project = {self.default_project_key} AND text ~ "{query}" ORDER BY updated DESC'
            else:
                jql = f'text ~ "{query}" ORDER BY updated DESC'

        if self.mock_mode:
            results = []
            q_lower = query.lower()
            for key, issue in self._mock_issues.items():
                if not query or (
                    q_lower in issue["key"].lower()
                    or q_lower in issue["summary"].lower()
                    or q_lower in issue["description"].lower()
                    or q_lower in issue["status"].lower()
                ):
                    results.append({
                        "key": issue["key"],
                        "summary": issue["summary"],
                        "status": issue["status"],
                        "priority": issue["priority"],
                        "issue_type": issue["issue_type"],
                        "assignee": issue["assignee"],
                        "updated": issue["updated"],
                        "url": f"https://mock-jira.atlassian.net/browse/{issue['key']}",
                    })
                if len(results) >= max_results:
                    break
            return results

        request_fields = fields or [
            "summary", "status", "priority", "issuetype", "assignee", "created", "updated"
        ]

        payload = {
            "jql": jql,
            "maxResults": min(max_results, 50),
            "fields": request_fields,
        }

        try:
            data = self._request("POST", "/rest/api/3/search/jql", data=payload)
        except JiraAPIError as e:
            if e.status_code in (404, 405, 410):
                try:
                    data = self._request("GET", "/rest/api/3/search/jql", params={"jql": jql, "maxResults": max_results})
                except JiraAPIError:
                    data = self._request("GET", "/rest/api/2/search", params={"jql": jql, "maxResults": max_results})
            else:
                raise

        issues = []
        for item in data.get("issues", []):
            f = item.get("fields", {})
            issues.append({
                "key": item.get("key"),
                "summary": f.get("summary", ""),
                "status": f.get("status", {}).get("name", "Unknown"),
                "priority": f.get("priority", {}).get("name", "None"),
                "issue_type": f.get("issuetype", {}).get("name", "Task"),
                "assignee": f.get("assignee", {}).get("displayName") if f.get("assignee") else "Unassigned",
                "updated": f.get("updated", ""),
                "url": f"{self.base_url}/browse/{item.get('key')}",
            })
        return issues

    # =========================================================================
    # 3. Get Issue
    # =========================================================================

    def get_issue(self, issue_key: str, include_comments: bool = True) -> Dict[str, Any]:
        """Retrieves full issue details, description, status, and recent comments."""
        issue_key = issue_key.strip().upper()

        if self.mock_mode:
            if issue_key not in self._mock_issues:
                raise JiraAPIError(f"Issue '{issue_key}' does not exist in mock database.", status_code=404)
            issue = dict(self._mock_issues[issue_key])
            if not include_comments:
                issue.pop("comments", None)
            return issue

        try:
            data = self._request("GET", f"/rest/api/3/issue/{issue_key}")
        except JiraAPIError as e:
            if e.status_code in (404, 410):
                data = self._request("GET", f"/rest/api/2/issue/{issue_key}")
            else:
                raise
        fields = data.get("fields", {})

        raw_desc = fields.get("description")
        description_text = adf_to_text(raw_desc).strip()

        comments_list = []
        if include_comments:
            raw_comments = fields.get("comment", {}).get("comments", [])
            for c in raw_comments[-10:]:
                c_body = adf_to_text(c.get("body")).strip()
                author_name = c.get("author", {}).get("displayName", "Unknown")
                comments_list.append({
                    "id": c.get("id"),
                    "author": author_name,
                    "created": c.get("created"),
                    "body": c_body,
                })

        return {
            "key": data.get("key"),
            "id": data.get("id"),
            "summary": fields.get("summary", ""),
            "description": description_text,
            "status": fields.get("status", {}).get("name", "Unknown"),
            "priority": fields.get("priority", {}).get("name", "None"),
            "issue_type": fields.get("issuetype", {}).get("name", "Task"),
            "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else "Unassigned",
            "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else "Unknown",
            "labels": fields.get("labels", []),
            "created": fields.get("created", ""),
            "updated": fields.get("updated", ""),
            "comments": comments_list,
            "url": f"{self.base_url}/browse/{data.get('key')}",
        }

    # =========================================================================
    # 4. Create Issue
    # =========================================================================

    def create_issue(
        self,
        summary: str,
        project_key: Optional[str] = None,
        issue_type: str = "Task",
        description: str = "",
        priority: str = "Medium",
        assignee: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Creates a new Jira issue. Uses project_key or defaults to JIRA_PROJECT_KEY."""
        self._check_read_only("create_issue")
        target_project = (project_key or self.default_project_key or "SCRUM").strip().upper()
        if not summary.strip():
            raise JiraAPIError("Issue summary cannot be empty.", status_code=400)

        if self.mock_mode:
            new_key = f"{target_project}-{self._mock_issue_counter}"
            self._mock_issue_counter += 1
            new_issue = {
                "key": new_key,
                "id": str(100000 + self._mock_issue_counter),
                "summary": summary,
                "description": description,
                "status": "To Do",
                "priority": priority,
                "issue_type": issue_type,
                "assignee": assignee or "Unassigned",
                "reporter": self.email or "Current User",
                "created": "2026-10-17T11:00:00.000+0000",
                "updated": "2026-10-17T11:00:00.000+0000",
                "labels": labels or [],
                "comments": [],
            }
            self._mock_issues[new_key] = new_issue
            return {
                "key": new_key,
                "id": new_issue["id"],
                "url": f"https://mock-jira.atlassian.net/browse/{new_key}",
                "message": f"Successfully created issue {new_key} in Mock Mode.",
            }

        fields: Dict[str, Any] = {
            "project": {"key": target_project},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }

        if priority:
            fields["priority"] = {"name": priority}

        if description:
            fields["description"] = description

        if labels:
            fields["labels"] = labels

        if assignee:
            if len(assignee) > 20 and "@" not in assignee:
                fields["assignee"] = {"accountId": assignee}
            else:
                fields["assignee"] = {"name": assignee}

        payload = {"fields": fields}

        try:
            res = self._request("POST", "/rest/api/2/issue", data=payload)
        except JiraAPIError as e:
            if "Atlassian Document Format" in str(e) or "ADF" in str(e):
                fields["description"] = text_to_adf(description)
                res = self._request("POST", "/rest/api/2/issue", data={"fields": fields})
            else:
                raise

        key = res.get("key")
        return {
            "key": key,
            "id": res.get("id"),
            "url": f"{self.base_url}/browse/{key}",
            "message": f"Successfully created issue {key}.",
        }

    # =========================================================================
    # 5. Update Issue
    # =========================================================================

    def update_issue(
        self,
        issue_key: str,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[str] = None,
        assignee: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Updates fields of an existing issue."""
        self._check_read_only("update_issue")
        issue_key = issue_key.strip().upper()

        if self.mock_mode:
            if issue_key not in self._mock_issues:
                raise JiraAPIError(f"Issue '{issue_key}' does not exist in mock database.", status_code=404)
            target = self._mock_issues[issue_key]
            updated_fields = []
            if summary is not None:
                target["summary"] = summary
                updated_fields.append("summary")
            if description is not None:
                target["description"] = description
                updated_fields.append("description")
            if priority is not None:
                target["priority"] = priority
                updated_fields.append("priority")
            if assignee is not None:
                target["assignee"] = assignee
                updated_fields.append("assignee")
            if labels is not None:
                target["labels"] = labels
                updated_fields.append("labels")
            target["updated"] = "2026-10-17T11:00:00.000+0000"
            return {
                "key": issue_key,
                "updated_fields": updated_fields,
                "message": f"Issue {issue_key} successfully updated in Mock Mode.",
            }

        fields: Dict[str, Any] = {}
        updated_fields = []
        if summary is not None:
            fields["summary"] = summary
            updated_fields.append("summary")
        if description is not None:
            fields["description"] = description
            updated_fields.append("description")
        if priority is not None:
            fields["priority"] = {"name": priority}
            updated_fields.append("priority")
        if assignee is not None:
            if len(assignee) > 20 and "@" not in assignee:
                fields["assignee"] = {"accountId": assignee}
            else:
                fields["assignee"] = {"name": assignee}
            updated_fields.append("assignee")
        if labels is not None:
            fields["labels"] = labels
            updated_fields.append("labels")

        if not fields:
            return {"key": issue_key, "updated_fields": [], "message": "No fields specified for update."}

        try:
            self._request("PUT", f"/rest/api/2/issue/{issue_key}", data={"fields": fields})
        except JiraAPIError as e:
            if "Atlassian Document Format" in str(e) or "ADF" in str(e):
                if "description" in fields:
                    fields["description"] = text_to_adf(description or "")
                    self._request("PUT", f"/rest/api/2/issue/{issue_key}", data={"fields": fields})
            else:
                raise

        return {
            "key": issue_key,
            "updated_fields": updated_fields,
            "message": f"Issue {issue_key} successfully updated.",
        }

    # =========================================================================
    # 6. Add Comment
    # =========================================================================

    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Adds a comment to a Jira issue."""
        self._check_read_only("add_comment")
        issue_key = issue_key.strip().upper()
        if not comment.strip():
            raise JiraAPIError("Comment body cannot be empty.", status_code=400)

        if self.mock_mode:
            if issue_key not in self._mock_issues:
                raise JiraAPIError(f"Issue '{issue_key}' does not exist in mock database.", status_code=404)
            new_id = str(len(self._mock_issues[issue_key]["comments"]) + 40001)
            comment_obj = {
                "id": new_id,
                "author": self.email or "Current User",
                "created": "2026-10-17T11:00:00.000+0000",
                "body": comment,
            }
            self._mock_issues[issue_key]["comments"].append(comment_obj)
            return {
                "id": new_id,
                "issue_key": issue_key,
                "message": f"Comment added to {issue_key} in Mock Mode.",
            }

        payload = {"body": comment}
        try:
            res = self._request("POST", f"/rest/api/2/issue/{issue_key}/comment", data=payload)
        except JiraAPIError as e:
            if "Atlassian Document Format" in str(e) or "ADF" in str(e):
                payload = {"body": text_to_adf(comment)}
                res = self._request("POST", f"/rest/api/2/issue/{issue_key}/comment", data=payload)
            else:
                raise

        return {
            "id": res.get("id"),
            "issue_key": issue_key,
            "author": res.get("author", {}).get("displayName"),
            "created": res.get("created"),
            "message": f"Comment successfully added to {issue_key}.",
        }

    # =========================================================================
    # 7. Transition Issue
    # =========================================================================

    def get_transitions(self, issue_key: str) -> List[Dict[str, Any]]:
        """Retrieves all valid workflow transitions available for an issue."""
        issue_key = issue_key.strip().upper()

        if self.mock_mode:
            if issue_key not in self._mock_issues:
                raise JiraAPIError(f"Issue '{issue_key}' does not exist in mock database.", status_code=404)
            current = self._mock_issues[issue_key]["status"]
            standard_transitions = [
                {"id": "11", "name": "To Do", "to_status": "To Do"},
                {"id": "21", "name": "In Progress", "to_status": "In Progress"},
                {"id": "31", "name": "Done", "to_status": "Done"},
            ]
            return [t for t in standard_transitions if t["to_status"].lower() != current.lower()]

        data = self._request("GET", f"/rest/api/2/issue/{issue_key}/transitions")
        transitions = []
        for t in data.get("transitions", []):
            transitions.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "to_status": t.get("to", {}).get("name", t.get("name")),
            })
        return transitions

    def transition_issue(
        self,
        issue_key: str,
        transition_name: Optional[str] = None,
        list_transitions: bool = False,
    ) -> Dict[str, Any]:
        """
        Transitions an issue to a new status (e.g. 'In Progress', 'Done').
        If list_transitions is True or transition_name is omitted, returns valid transitions.
        """
        issue_key = issue_key.strip().upper()
        available = self.get_transitions(issue_key)

        if list_transitions or not transition_name:
            return {
                "issue_key": issue_key,
                "available_transitions": available,
                "message": f"Found {len(available)} available transition(s) for {issue_key}.",
            }

        self._check_read_only("transition_issue")
        target_name = transition_name.strip().lower()
        matched = None
        for t in available:
            if (
                t["name"].lower() == target_name
                or t["to_status"].lower() == target_name
                or t["id"] == transition_name.strip()
            ):
                matched = t
                break

        if not matched:
            valid_names = [f"'{t['name']}' (to {t['to_status']})" for t in available]
            raise JiraAPIError(
                f"Transition '{transition_name}' is not valid for issue {issue_key}. "
                f"Available transitions are: {', '.join(valid_names)}.",
                status_code=400,
            )

        if self.mock_mode:
            old_status = self._mock_issues[issue_key]["status"]
            self._mock_issues[issue_key]["status"] = matched["to_status"]
            self._mock_issues[issue_key]["updated"] = "2026-10-17T11:00:00.000+0000"
            return {
                "issue_key": issue_key,
                "previous_status": old_status,
                "new_status": matched["to_status"],
                "transition_applied": matched["name"],
                "message": f"Issue {issue_key} transitioned from '{old_status}' to '{matched['to_status']}' in Mock Mode.",
            }

        payload = {"transition": {"id": matched["id"]}}
        self._request("POST", f"/rest/api/2/issue/{issue_key}/transitions", data=payload)

        return {
            "issue_key": issue_key,
            "new_status": matched["to_status"],
            "transition_applied": matched["name"],
            "message": f"Issue {issue_key} successfully transitioned to '{matched['to_status']}'.",
        }
