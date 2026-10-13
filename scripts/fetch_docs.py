"""Fetch NestJS documentation markdown files from official GitHub repository."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_API_BASE = "https://api.github.com/repos/nestjs/docs.nestjs.com/contents/content"
RAW_BASE = "https://raw.githubusercontent.com/nestjs/docs.nestjs.com/master/content"
DATA_DIR = Path("data/nestjs")

CORE_SECTIONS = [
    "",  # root content/
    "fundamentals",
    "security",
    "techniques",
    "microservices",
    "websockets",
    "recipes",
]

CORE_FALLBACK_FILES = [
    "first-steps.md",
    "controllers.md",
    "providers.md",
    "modules.md",
    "middlewares.md",
    "exception-filters.md",
    "pipes.md",
    "guards.md",
    "interceptors.md",
    "custom-decorators.md",
    "application-context.md",
]


def fetch_file_content(relative_path: str) -> str:
    """Fetch raw markdown file content from GitHub raw endpoint."""
    url = f"{RAW_BASE}/{relative_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "DevDocs-Copilot-Fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


def list_directory_files(section: str) -> list[str]:
    """List markdown files in a section using GitHub API, or return fallback."""
    api_url = f"{REPO_API_BASE}/{section}" if section else REPO_API_BASE
    req = urllib.request.Request(api_url, headers={"User-Agent": "DevDocs-Copilot-Fetcher/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read().decode("utf-8"))
            return [
                f"{section}/{item['name']}" if section else item["name"]
                for item in items
                if item["type"] == "file" and item["name"].endswith(".md")
            ]
    except urllib.error.HTTPError as err:
        print(f"Warning: API listing failed for section '{section}' ({err}). Using fallback list.")
        if not section:
            return CORE_FALLBACK_FILES
        return []


def main() -> None:
    """Download markdown files and save them locally in DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching NestJS documentation into {DATA_DIR}...")

    all_files: list[str] = []
    for section in CORE_SECTIONS:
        section_files = list_directory_files(section)
        all_files.extend(section_files)

    if not all_files:
        all_files = CORE_FALLBACK_FILES

    # Deduplicate while preserving order
    unique_files = list(dict.fromkeys(all_files))
    print(f"Identified {len(unique_files)} markdown files to download.")

    downloaded = 0
    for rel_path in unique_files:
        filename = rel_path.replace("/", "__")
        dest_path = DATA_DIR / filename
        if dest_path.exists() and dest_path.stat().st_size > 0:
            downloaded += 1
            continue

        try:
            print(f"Downloading {rel_path} -> {filename}...")
            content = fetch_file_content(rel_path)
            dest_path.write_text(content, encoding="utf-8")
            downloaded += 1
        except Exception as exc:
            print(f"Failed to fetch {rel_path}: {exc}")

    print(f"Successfully prepared {downloaded} documentation files in {DATA_DIR}.")


if __name__ == "__main__":
    main()
