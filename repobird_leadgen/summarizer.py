import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

# Removed unused imports: datetime, csv, Table, RepoSummary
# from .models import EnrichedRepoData # Not strictly needed here

console = Console()

# Removed _format_output_path as the filename is now determined in cli.py
# Removed to_markdown, to_jsonl (old version), to_csv


def save_enriched_data(
    enriched_data: List[Dict[str, Any]], outfile: Path
) -> None:
    """
    Saves the enriched repository data (including issue analysis) to a JSON Lines file.

    Args:
        enriched_data: A list of dictionaries, where each dictionary represents
                       an enriched repository's data (output from enrich_repo_entry).
        outfile: The Path object representing the output JSONL file.
    """
    if not enriched_data:
        console.print("[yellow]No enriched data provided, skipping output file creation.")
        return

    try:
        outfile.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with outfile.open("w", encoding="utf-8") as f:
            for item in enriched_data:
                # Ensure complex objects (like datetime if they were present) are serializable
                # In our current structure (dict + list of dicts), standard json dump should work
                # If IssueAnalysis dataclass instances were passed directly, we'd need asdict() here.
                # Since enrich_repo_entry already converts to dict, this is fine.
                f.write(json.dumps(item) + "\n")
        console.print(f"[green]Saved enriched data (JSONL) → {outfile}")
    except Exception as e:
        console.print(f"[red]Error writing enriched JSONL file {outfile}: {e}")

# Removed the old save_summary function
