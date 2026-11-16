"""
Push defect tickets from defectdump.json to a live Jira Cloud workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


def load_config() -> dict[str, str]:
    # Look for .env in current directory, script parent directory, or workspace root
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent / "Defect Triage" / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for p in env_paths:
        if p.is_file():
            load_dotenv(p)
            break

    base_url = (os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_URL") or "").strip().rstrip("/")
    email = (os.getenv("JIRA_EMAIL") or "").strip()
    api_token = (os.getenv("JIRA_API_TOKEN") or "").strip()
    project_key = (os.getenv("JIRA_PROJECT_KEY") or "").strip()

    return {
        "base_url": base_url,
        "email": email,
        "api_token": api_token,
        "project_key": project_key,
    }


def validate_credentials(config: dict[str, str]) -> tuple[bool, str]:
    placeholder_domains = ["your-domain.atlassian.net", "your-company.atlassian.net", "example.com"]
    if not config["base_url"] or any(d in config["base_url"] for d in placeholder_domains):
        return False, "JIRA_BASE_URL is missing or still set to a placeholder in .env"
    if not config["email"] or "example.com" in config["email"]:
        return False, "JIRA_EMAIL is missing or still set to a placeholder in .env"
    if not config["api_token"] or "your" in config["api_token"].lower():
        return False, "JIRA_API_TOKEN is missing or still set to a placeholder in .env"
    return True, ""


def get_jira_session(config: dict[str, str]) -> requests.Session:
    session = requests.Session()
    session.auth = (config["email"], config["api_token"])
    session.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    return session


def verify_connection(session: requests.Session, base_url: str) -> dict[str, Any]:
    url = f"{base_url}/rest/api/3/myself"
    response = session.get(url, timeout=15)
    if response.status_code == 401:
        raise RuntimeError("Authentication failed (401 Unauthorized). Please check your JIRA_EMAIL and JIRA_API_TOKEN.")
    response.raise_for_status()
    return response.json()


def get_available_projects(session: requests.Session, base_url: str) -> list[dict[str, Any]]:
    url = f"{base_url}/rest/api/3/project"
    response = session.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def get_issue_types_for_project(session: requests.Session, base_url: str, project_id_or_key: str) -> list[str]:
    url = f"{base_url}/rest/api/3/project/{project_id_or_key}"
    response = session.get(url, timeout=15)
    if response.status_code == 200:
        proj_data = response.json()
        return [it["name"] for it in proj_data.get("issueTypes", [])]
    return []


def create_jira_issue(
    session: requests.Session,
    base_url: str,
    project_key: str,
    issue_type: str,
    summary: str,
    description_text: str,
    priority: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    # Build Atlassian Document Format (ADF) description
    paragraphs = []
    for line in description_text.split("\n\n"):
        clean_line = line.strip()
        if clean_line:
            paragraphs.append({
                "type": "paragraph",
                "content": [{"type": "text", "text": clean_line}],
            })

    if not paragraphs:
        paragraphs.append({
            "type": "paragraph",
            "content": [{"type": "text", "text": summary}],
        })

    fields: dict[str, Any] = {
        "project": {"key": project_key},
        "summary": summary,
        "description": {
            "type": "doc",
            "version": 1,
            "content": paragraphs,
        },
        "issuetype": {"name": issue_type},
    }

    if priority:
        fields["priority"] = {"name": priority}

    if labels:
        # Jira labels must not contain spaces
        sanitized_labels = [
            label.replace(" ", "_").replace(":", "").replace("[", "").replace("]", "")[:255]
            for label in labels
            if label.strip()
        ]
        if sanitized_labels:
            fields["labels"] = sanitized_labels

    url = f"{base_url}/rest/api/3/issue"
    response = session.post(url, json={"fields": fields}, timeout=30)
    
    if response.status_code == 400 and priority:
        # Retry without priority in case the project doesn't have matching priority scheme
        del fields["priority"]
        response = session.post(url, json={"fields": fields}, timeout=30)

    if not response.ok:
        err_detail = response.text
        try:
            err_json = response.json()
            err_detail = json.dumps(err_json.get("errors") or err_json.get("errorMessages") or err_json)
        except Exception:
            pass
        raise RuntimeError(f"HTTP {response.status_code}: {err_detail}")

    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Push defectdump.json tickets to Jira.")
    parser.add_argument("--file", default="defectdump.json", help="Path to defectdump.json")
    parser.add_argument("--project-key", help="Target Jira project key (overrides .env and json)")
    parser.add_argument("--issue-type", default="Bug", help="Issue type name (default: Bug, fallback: Task)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickets to push")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index in the JSON to start pushing from")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip tickets whose summary already exists in the project")
    parser.add_argument("--dry-run", action="store_true", help="Preview tickets without pushing")
    args = parser.parse_args()

    config = load_config()
    is_valid, reason = validate_credentials(config)

    file_path = Path(args.file)
    if not file_path.is_file():
        # Check relative to script
        file_path = Path(__file__).resolve().parent / args.file
    if not file_path.is_file():
        print(f"[-] Defect file not found: {args.file}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        tickets: list[dict[str, Any]] = json.load(f)

    if args.limit:
        tickets = tickets[: args.limit]

    print("=" * 60)
    print("  JIRA DEFECT TICKET PUSHER")
    print("=" * 60)
    print(f"[+] Loaded {len(tickets)} tickets from: {file_path.name}")

    if args.dry_run:
        print("[!] Running in DRY-RUN mode. No tickets will be created in Jira.")
        for idx, t in enumerate(tickets, 1):
            pk = args.project_key or config["project_key"] or t.get("project_key", "FS")
            print(f"  {idx}. [{pk}] {t.get('summary')} (Priority: {t.get('priority')})")
        print("\n[+] Dry run complete.")
        return

    if not is_valid:
        print(f"\n[-] Cannot push to Jira yet: {reason}")
        print("\nPlease update your credentials in:")
        print(f"  {Path(__file__).resolve().parent / '.env'}\n")
        print("Required fields:")
        print("  JIRA_BASE_URL=https://your-company.atlassian.net")
        print("  JIRA_EMAIL=your-email@example.com")
        print("  JIRA_API_TOKEN=your-atlassian-api-token")
        print("  JIRA_PROJECT_KEY=your-project-key  (e.g., SCRUM or KAN)")
        sys.exit(1)

    print(f"[+] Connecting to Jira instance: {config['base_url']}")
    session = get_jira_session(config)

    try:
        user_info = verify_connection(session, config["base_url"])
        print(f"[+] Authenticated as: {user_info.get('displayName')} ({user_info.get('emailAddress')})")
    except Exception as e:
        print(f"[-] Connection test failed: {e}")
        sys.exit(1)

    # Check available projects
    try:
        projects = get_available_projects(session, config["base_url"])
        proj_keys = [p["key"] for p in projects]
        print(f"[+] Available projects in your Jira workspace: {', '.join(proj_keys)}")
    except Exception:
        proj_keys = []

    target_project = args.project_key or config["project_key"]
    if not target_project:
        if len(proj_keys) == 1:
            target_project = proj_keys[0]
            print(f"[i] Auto-selected the only available project: {target_project}")
        elif "FS" in proj_keys:
            target_project = "FS"
        else:
            print(f"\n[-] Please specify a target Jira project key with --project-key <KEY> or set JIRA_PROJECT_KEY in .env.")
            print(f"    Available projects: {', '.join(proj_keys)}")
            sys.exit(1)

    print(f"[+] Target project key: {target_project}")

    # Check issue type
    issue_types = get_issue_types_for_project(session, config["base_url"], target_project)
    issue_type = args.issue_type
    if issue_types and issue_type not in issue_types:
        if "Bug" in issue_types:
            issue_type = "Bug"
        elif "Task" in issue_types:
            issue_type = "Task"
        else:
            issue_type = issue_types[0]
        print(f"[i] Using available issue type: {issue_type}")

    existing_summaries = set()
    if args.skip_existing:
        try:
            search_res = session.post(
                f"{config['base_url']}/rest/api/3/search/jql",
                json={"jql": f"project = {target_project}", "maxResults": 100, "fields": ["summary"]},
                timeout=20,
            )
            if search_res.status_code == 200:
                for issue in search_res.json().get("issues", []):
                    s = issue.get("fields", {}).get("summary")
                    if s:
                        existing_summaries.add(s.strip())
                if existing_summaries:
                    print(f"[i] Found {len(existing_summaries)} existing ticket(s) in {target_project} to skip.")
        except Exception as e:
            print(f"[!] Warning: Could not check existing tickets: {e}")

    print(f"\n[+] Starting push of {len(tickets)} tickets (starting from #{args.start_index})...")
    created_keys: list[str] = []

    for idx, t in enumerate(tickets, 1):
        if idx < args.start_index:
            continue

        summary = t.get("summary", "Defect")
        if args.skip_existing and summary.strip() in existing_summaries:
            print(f"  [{idx}/{len(tickets)}] Skipping already existing: '{summary[:45]}...'")
            continue
        raw_desc = t.get("description", "")
        labels = t.get("labels", [])
        priority = t.get("priority")

        # Format full description with logs & root cause preserved
        desc_parts = [raw_desc]
        clean_tags = ["defect_dump"]
        for lbl in labels:
            if lbl.lower().startswith("logs:"):
                desc_parts.append(f"**Logs:**\n{lbl[5:].strip()}")
            elif lbl.lower().startswith("root_cause:"):
                desc_parts.append(f"**Root Cause:**\n{lbl[11:].strip()}")
            else:
                clean_tags.append(lbl)

        full_desc = "\n\n".join(desc_parts)

        try:
            res = create_jira_issue(
                session=session,
                base_url=config["base_url"],
                project_key=target_project,
                issue_type=issue_type,
                summary=summary,
                description_text=full_desc,
                priority=priority,
                labels=clean_tags,
            )
            key = res.get("key", "UNKNOWN")
            created_keys.append(key)
            print(f"  [{idx}/{len(tickets)}] Created ticket {key}: {summary[:50]}...")
        except Exception as e:
            print(f"  [{idx}/{len(tickets)}] Error creating ticket '{summary[:30]}...': {e}")

    print("\n" + "=" * 60)
    print(f"  COMPLETED: {len(created_keys)}/{len(tickets)} tickets pushed successfully!")
    print("=" * 60)
    if created_keys:
        print(f"Ticket range created: {created_keys[0]} to {created_keys[-1]}")
        print("You can now run Defect_Triaging.py or jira-defect-embedding.py against your live Jira workspace!")


if __name__ == "__main__":
    main()
