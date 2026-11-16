"""
Automated Test & Verification Suite for Smart Jira MCP Server
Validates:
1. JiraClient unit methods (projects, search, get, create, update, comment, transition).
2. Code-level read-only security enforcement (JIRA_READ_ONLY=true blocking mutations).
3. ChromaDB Vector Similarity Search on synthetic defects.
4. AI Defect Triage recommendation logic.
5. End-to-end MCP JSON-RPC protocol over stdio (handshake, 9 tool discovery, tool invocation).
"""

import os
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from jira_client import JiraClient, JiraAPIError, text_to_adf, adf_to_text
from defect_triage_engine import search_similar_defects, build_triage_prompt
from mcp_client import JiraMCPClient


class TestJiraClientUnit(unittest.TestCase):
    """Unit tests for JiraClient logic and ADF conversions (read_only=False)."""

    def setUp(self):
        # Explicitly permit writes for unit lifecycle testing
        self.client = JiraClient(mock_mode=True, read_only=False)

    def test_adf_and_text_conversion(self):
        sample_text = "Line 1\nLine 2\n\nParagraph 2 with text"
        adf = text_to_adf(sample_text)
        self.assertEqual(adf["type"], "doc")
        self.assertTrue(len(adf["content"]) >= 2)
        recovered = adf_to_text(adf)
        self.assertIn("Line 1", recovered)
        self.assertIn("Paragraph 2", recovered)

    def test_list_projects(self):
        projects = self.client.list_projects()
        self.assertTrue(len(projects) >= 1)
        keys = [p["key"] for p in projects]
        self.assertIn("SCRUM", keys)

    def test_search_issues(self):
        results = self.client.search_issues("token", max_results=5)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["key"], "SCRUM-1")

    def test_get_issue(self):
        issue = self.client.get_issue("SCRUM-1", include_comments=True)
        self.assertEqual(issue["key"], "SCRUM-1")
        self.assertEqual(issue["status"], "In Progress")
        self.assertTrue(len(issue["comments"]) >= 1)

    def test_create_and_update_issue(self):
        created = self.client.create_issue(
            summary="Automated test bug",
            issue_type="Bug",
            description="Created during automated tests",
            priority="Low",
            labels=["test", "automation"],
        )
        new_key = created["key"]
        self.assertTrue(new_key.startswith("SCRUM-"))

        updated = self.client.update_issue(
            issue_key=new_key,
            summary="Updated automated issue summary",
            priority="High",
        )
        self.assertIn("summary", updated["updated_fields"])
        self.assertIn("priority", updated["updated_fields"])

        issue = self.client.get_issue(new_key)
        self.assertEqual(issue["summary"], "Updated automated issue summary")
        self.assertEqual(issue["priority"], "High")

    def test_add_comment(self):
        comment_res = self.client.add_comment("SCRUM-2", "Investigating backend latency.")
        self.assertIn("id", comment_res)

        issue = self.client.get_issue("SCRUM-2", include_comments=True)
        bodies = [c["body"] for c in issue["comments"]]
        self.assertIn("Investigating backend latency.", bodies)

    def test_transition_issue(self):
        transitions = self.client.get_transitions("SCRUM-2")
        trans_names = [t["name"] for t in transitions]
        self.assertIn("In Progress", trans_names)

        res = self.client.transition_issue("SCRUM-2", transition_name="In Progress")
        self.assertEqual(res["new_status"], "In Progress")

        issue = self.client.get_issue("SCRUM-2")
        self.assertEqual(issue["status"], "In Progress")


class TestReadOnlySecurity(unittest.TestCase):
    """
    Validates code-level read-only security enforcement (JIRA_READ_ONLY=true).
    Guarantees that no mutation occurs when read-only mode is active.
    """

    def setUp(self):
        # Client initialized in strict read-only mode
        self.ro_client = JiraClient(mock_mode=True, read_only=True)

    def test_read_only_blocks_create_issue(self):
        with self.assertRaises(JiraAPIError) as ctx:
            self.ro_client.create_issue(summary="Blocked bug")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("blocked", str(ctx.exception).lower())

    def test_read_only_blocks_update_issue(self):
        with self.assertRaises(JiraAPIError) as ctx:
            self.ro_client.update_issue("SCRUM-1", summary="New summary")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("blocked", str(ctx.exception).lower())

    def test_read_only_blocks_add_comment(self):
        with self.assertRaises(JiraAPIError) as ctx:
            self.ro_client.add_comment("SCRUM-1", "Blocked comment")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("blocked", str(ctx.exception).lower())

    def test_read_only_blocks_transition_issue(self):
        with self.assertRaises(JiraAPIError) as ctx:
            self.ro_client.transition_issue("SCRUM-1", transition_name="Done")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("blocked", str(ctx.exception).lower())

    def test_read_only_allows_read_operations(self):
        # Reading projects, searching, fetching details, and listing transitions remain allowed
        projects = self.ro_client.list_projects()
        self.assertTrue(len(projects) >= 1)

        issues = self.ro_client.search_issues("SCRUM-1")
        self.assertTrue(len(issues) >= 1)

        issue = self.ro_client.get_issue("SCRUM-1")
        self.assertEqual(issue["key"], "SCRUM-1")

        trans = self.ro_client.transition_issue("SCRUM-1", list_transitions=True)
        self.assertIn("available_transitions", trans)


class TestDefectTriageEngine(unittest.TestCase):
    """Unit tests for ChromaDB vector search and prompt builder."""

    def test_chroma_semantic_search(self):
        matches = search_similar_defects("Payment gateway timeout during peak flash sale", top_k=2)
        self.assertTrue(len(matches) >= 1)
        self.assertIn("similarity", matches[0])
        self.assertIn("root_cause", matches[0])
        self.assertTrue(matches[0]["similarity"] > 0.0)

    def test_build_triage_prompt(self):
        matches = [{"defect_id": "SCRUM-99", "similarity": 0.88, "document": "Sample error", "root_cause": "OOM"}]
        prompt = build_triage_prompt("SCRUM-1", "Test defect", "Test description", matches)
        self.assertIn("SCRUM-1", prompt)
        self.assertIn("SCRUM-99", prompt)
        self.assertIn("OOM", prompt)


class TestSmartJiraMCPServerStdio(unittest.TestCase):
    """End-to-end tests connecting to the Smart Jira MCP Server via stdio."""

    @classmethod
    def setUpClass(cls):
        os.environ["JIRA_MOCK_MODE"] = "true"
        os.environ["JIRA_READ_ONLY"] = "true"
        cls.client = JiraMCPClient(server_path=BASE_DIR / "server.py")
        cls.client.start()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_tool_discovery_9_tools(self):
        tools = self.client.list_tools()
        tool_names = [t.name for t in tools]
        expected_tools = [
            "jira_list_projects",
            "jira_search_issues",
            "jira_get_issue",
            "jira_create_issue",
            "jira_update_issue",
            "jira_add_comment",
            "jira_transition_issue",
            "jira_find_similar_defects",
            "jira_triage_defect",
        ]
        for expected in expected_tools:
            self.assertIn(expected, tool_names, f"Tool {expected} not found in MCP server!")
        self.assertEqual(len(tool_names), 9, f"Expected 9 tools, found {len(tool_names)}")

    def test_mcp_call_find_similar_defects(self):
        res = self.client.call_tool("jira_find_similar_defects", {
            "query": "Database connection drop during high query load",
            "top_k": 2,
        })
        self.assertFalse(res.is_error)
        data = res.data
        self.assertIn("matches", data)
        self.assertTrue(len(data["matches"]) >= 1)

    def test_mcp_call_triage_defect(self):
        res = self.client.call_tool("jira_triage_defect", {
            "issue_key": "SCRUM-1",
            "post_comment_to_jira": False,
        })
        self.assertFalse(res.is_error)
        data = res.data
        self.assertEqual(data.get("issue_key"), "SCRUM-1")
        self.assertIn("recommendation", data)

    def test_mcp_stdio_read_only_blocks_create(self):
        res = self.client.call_tool("jira_create_issue", {
            "summary": "Attempted write via MCP in read-only mode",
        })
        data = res.data
        self.assertTrue(data.get("read_only") or "blocked" in data.get("error", "").lower())

    def test_mcp_stdio_read_only_blocks_triage_comment(self):
        res = self.client.call_tool("jira_triage_defect", {
            "issue_key": "SCRUM-1",
            "post_comment_to_jira": True,
        })
        data = res.data
        self.assertTrue(data.get("read_only") or "blocked" in data.get("error", "").lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
