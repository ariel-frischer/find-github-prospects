from datetime import datetime
from typing import List
from pathlib import Path
import json
import csv # Add CSV support
from rich.console import Console
from rich.table import Table # Use Rich for better table rendering
from .models import RepoSummary, ContactInfo

console = Console()

def _format_output_path(outdir: Path, base_name: str, extension: str) -> Path:
    """Creates a timestamped output path."""
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{base_name}_{ts}.{extension}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / filename

def to_markdown(summaries: List[RepoSummary], outfile: Path) -> None:
    """Pretty‑print a Markdown table of prospects."""
    if not summaries:
        console.print("[yellow]No summaries to generate Markdown for.")
        return

    table = Table(title=f"RepoBird LeadGen Prospects ({outfile.stem})", show_header=True, header_style="bold magenta")
    headers = [
        "Repo", "Stars", "Lang", "Open Issues", "Good First", "Help Wanted", "Last Push", "Emails", "Twitter", "Blog"
    ]
    for header in headers:
        table.add_column(header)

    for s in summaries:
        table.add_row(
            f"[{s.full_name}](https://github.com/{s.full_name})",
            str(s.stars),
            s.language or "-",
            str(s.open_issues),
            str(s.good_first_issues),
            str(s.help_wanted_issues),
            s.last_push.strftime('%Y-%m-%d') if s.last_push else "-", # Handle potential None
            ", ".join(s.contact.emails) if s.contact.emails else "-",
            f"@{s.contact.twitter}" if s.contact.twitter else "-",
            s.contact.blog if s.contact.blog else "-",
        )

    # Instead of writing raw Markdown, let Rich render the table to the console
    # and save a simpler Markdown version to the file.
    console.print(table)

    # Generate Markdown text for the file
    md_content = (
        f"# RepoBird LeadGen Prospects ({outfile.stem})\n\n"
    )
    md_content += "| " + " | ".join(headers) + " |\n"
    md_content += "|" + "---|" * len(headers) + "\n"
    for s in summaries:
         row_data = [
             f"[{s.full_name}](https://github.com/{s.full_name})",
             str(s.stars),
             s.language or "-",
             str(s.open_issues),
             str(s.good_first_issues),
             str(s.help_wanted_issues),
             s.last_push.strftime('%Y-%m-%d') if s.last_push else "-",
             "`" + ", ".join(s.contact.emails) + "`" if s.contact.emails else "-", # Code format emails
             f"[@{s.contact.twitter}](https://twitter.com/{s.contact.twitter})" if s.contact.twitter else "-",
             f"[Blog]({s.contact.blog})" if s.contact.blog else "-",
         ]
         md_content += "| " + " | ".join(row_data) + " |\n" # Added closing quote

    try:
        outfile.write_text(md_content, encoding='utf-8')
        console.print(f"[green]Saved Markdown prospect list → {outfile}")
    except Exception as e:
        console.print(f"[red]Error writing Markdown file {outfile}: {e}")


def to_jsonl(summaries: List[RepoSummary], outfile: Path) -> None:
    """Save summaries as JSON Lines."""
    if not summaries:
        console.print("[yellow]No summaries to generate JSONL for.")
        return

    try:
        with outfile.open("w", encoding='utf-8') as f:
            for s in summaries:
                # Convert dataclass to dict, handling datetime
                summary_dict = s.__dict__.copy()
                summary_dict['last_push'] = s.last_push.isoformat() if s.last_push else None
                summary_dict['contact'] = s.contact.__dict__ # Convert nested dataclass too
                f.write(json.dumps(summary_dict) + "
")
        console.print(f"[green]Saved JSONL → {outfile}")
    except Exception as e:
        console.print(f"[red]Error writing JSONL file {outfile}: {e}")

def to_csv(summaries: List[RepoSummary], outfile: Path) -> None:
    """Save summaries as CSV."""
    if not summaries:
        console.print("[yellow]No summaries to generate CSV for.")
        return

    headers = [
        "full_name", "description", "stars", "language", "open_issues",
        "good_first_issues", "help_wanted_issues", "last_push",
        "contact_emails", "contact_twitter", "contact_blog"
    ]
    try:
        with outfile.open("w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for s in summaries:
                writer.writerow({
                    "full_name": s.full_name,
                    "description": s.description,
                    "stars": s.stars,
                    "language": s.language or "",
                    "open_issues": s.open_issues,
                    "good_first_issues": s.good_first_issues,
                    "help_wanted_issues": s.help_wanted_issues,
                    "last_push": s.last_push.strftime('%Y-%m-%d %H:%M:%S') if s.last_push else "",
                    "contact_emails": ",".join(s.contact.emails),
                    "contact_twitter": s.contact.twitter or "",
                    "contact_blog": s.contact.blog or "",
                })
        console.print(f"[green]Saved CSV → {outfile}")
    except Exception as e:
        console.print(f"[red]Error writing CSV file {outfile}: {e}")


def save_summary(summaries: List[RepoSummary], outdir_base: str, formats: List[str]) -> None:
    """Saves the repo summaries in the requested formats."""
    outdir = Path(outdir_base)
    if not summaries:
         console.print("[yellow]No summaries generated, skipping output file creation.")
         return

    if "md" in formats or "markdown" in formats:
        md_path = _format_output_path(outdir, "prospects", "md")
        to_markdown(summaries, md_path)
    if "jsonl" in formats:
        jsonl_path = _format_output_path(outdir, "prospects", "jsonl")
        to_jsonl(summaries, jsonl_path)
    if "csv" in formats:
         csv_path = _format_output_path(outdir, "prospects", "csv")
         to_csv(summaries, csv_path)

