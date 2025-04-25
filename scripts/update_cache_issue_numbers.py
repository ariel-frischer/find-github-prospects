#!/usr/bin/env python
import json
import shutil
import tempfile
import time  # Import the time module
from pathlib import Path
from typing import Dict, Any  # Keep List here

import typer
from rich.progress import track
from rich import print

# Assume repobird_leadgen is installed or PYTHONPATH is set correctly
from repobird_leadgen.browser_checker import BrowserIssueChecker

# Script to fix our cached old jsonl items that were created before the _repobird_found_issue_numbers field was needed.
app = typer.Typer(
    add_completion=False,
    help="Updates a raw repo cache JSONL file by adding missing '_repobird_found_issues' or upgrading existing integer-only lists to full issue details using the BrowserIssueChecker.",
    rich_markup_mode="markdown",
)


@app.command()
def main(
    cache_file: Path = typer.Argument(
        ...,
        help="Path to the raw_repos_*.jsonl cache file to update.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=True,
    ),
    label: str = typer.Option(
        ...,
        "--label",
        "-l",
        help="The issue label that was used to generate this cache file (required).",
    ),
    headless: bool = typer.Option(
        True,
        "--headless/--no-headless",
        help="Run the browser checker in headless mode.",
    ),
    delay: float = typer.Option(
        2.0, "--delay", "-d", help="Delay (seconds) between browser checks."
    ),
    timeout: int = typer.Option(
        45000, "--timeout", "-t", help="Timeout (milliseconds) for browser actions."
    ),
):
    """
    Processes a JSONL cache file, finds lines missing the issue details field
    or containing only issue numbers, runs the browser checker to get full details,
    and updates the file in place.
    """
    print(
        f"[bold blue]Updating/Upgrading cache file with issue details:[/bold blue] {cache_file}"
    )
    print(f"  Using label: '{label}' for checks")
    print(f"  Browser Headless: {headless}")

    updated_count = 0
    processed_count = 0
    error_count = 0

    # Create a temporary file in the same directory to handle potential move issues across filesystems
    temp_file_path = None
    try:
        with (
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=cache_file.parent,
                prefix=f"{cache_file.stem}_update_",
                suffix=".jsonl",
            ) as temp_f,
            cache_file.open("r", encoding="utf-8") as input_f,
            BrowserIssueChecker(
                headless=headless, check_delay=delay, timeout=timeout
            ) as checker,
        ):
            temp_file_path = Path(temp_f.name)
            print(f"Using temporary file: {temp_file_path}")

            # Estimate number of lines for progress bar (optional but nice)
            try:
                num_lines = sum(1 for _ in input_f)
                input_f.seek(0)  # Reset file pointer
            except Exception:
                num_lines = None
                print("[yellow]Could not determine file size for progress bar.")

            # Process lines with progress tracking
            for line in track(
                input_f, description="Processing cache...", total=num_lines
            ):
                processed_count += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data: Dict[str, Any] = json.loads(line)
                    repo_name = data.get("full_name")

                    if not repo_name:
                        print(
                            f"[yellow]Skipping line {processed_count}: Missing 'full_name'."
                        )
                        temp_f.write(line + "\n")  # Write original line back
                        continue

                    needs_update = False
                    issue_numbers_to_fetch = []

                    if "_repobird_found_issues" not in data:
                        needs_update = True
                        print(
                            f"\n[cyan]Field missing, fetching issue details for:[/cyan] {repo_name}"
                        )
                        # Use check_repo_for_issue_label which gets numbers and details
                        try:
                            has_label, issue_details = (
                                checker.check_repo_for_issue_label(repo_name, label)
                            )
                            data["_repobird_found_issues"] = (
                                issue_details  # Add the new field
                            )
                            print(
                                f"  [green]Added issue details:[/green] {len(issue_details)} issues found (Has Label: {has_label})"
                            )
                            data.pop(
                                "_repobird_found_issue_numbers", None
                            )  # Clean up old field if present
                            updated_count += 1
                        except Exception as check_err:
                            print(
                                f"[red]Error checking issues for {repo_name}: {check_err}. Skipping update."
                            )
                            error_count += 1
                            needs_update = False  # Prevent writing potentially bad data
                            temp_f.write(line + "\n")  # Write original line back
                            continue  # Skip to next line

                    elif isinstance(data["_repobird_found_issues"], list) and all(
                        isinstance(item, int) for item in data["_repobird_found_issues"]
                    ):
                        # Field exists but contains only integers (old format)
                        needs_update = True
                        issue_numbers_to_fetch = data["_repobird_found_issues"]
                        print(
                            f"\n[cyan]Old format found, fetching details for {len(issue_numbers_to_fetch)} issues in:[/cyan] {repo_name}"
                        )
                        # Fetch details individually using the existing numbers
                        new_issue_details = []
                        temp_page = None  # Use a single page for this repo's issues
                        try:
                            temp_page = (
                                checker._get_new_page()
                            )  # Get a page from the checker
                            for issue_num in issue_numbers_to_fetch:
                                issue_url = (
                                    f"https://github.com/{repo_name}/issues/{issue_num}"
                                )
                                details = checker._fetch_issue_details(
                                    temp_page, issue_url
                                )
                                if details:
                                    new_issue_details.append(
                                        {
                                            "number": issue_num,
                                            "title": details["title"],
                                            "description": details["description"],
                                        }
                                    )
                                else:
                                    print(
                                        f"    [yellow]Skipping details for issue {issue_num} due to fetch error."
                                    )
                                time.sleep(
                                    0.5
                                )  # Small delay between issue detail fetches

                            data["_repobird_found_issues"] = (
                                new_issue_details  # Replace old list with new details
                            )
                            print(
                                f"  [green]Upgraded issue details:[/green] Fetched details for {len(new_issue_details)} issues."
                            )
                            updated_count += 1
                        except Exception as fetch_err:
                            print(
                                f"[red]Error fetching individual issue details for {repo_name}: {fetch_err}. Skipping update."
                            )
                            error_count += 1
                            needs_update = False  # Prevent writing potentially bad data
                            temp_f.write(line + "\n")  # Write original line back
                            continue  # Skip to next line
                        finally:
                            if temp_page:
                                try:
                                    temp_page.close()
                                except Exception:
                                    pass  # Ignore errors closing temp page

                    elif isinstance(data["_repobird_found_issues"], list):
                        # Field exists and is a list, assume it's already the new format (list of dicts)
                        # print(f"Skipping line {processed_count}: Already has issue details.") # Optional: uncomment for verbose logging
                        pass  # No update needed
                    else:
                        # Field exists but is not a list (unexpected format)
                        print(
                            f"[yellow]Skipping line {processed_count}: Unexpected format for _repobird_found_issues: {type(data['_repobird_found_issues'])}"
                        )
                        needs_update = False  # Don't attempt to write

                    # Write the (potentially updated) data back to the temp file if an update happened or no update was needed
                    # Only skip writing if an error occurred during fetching/updating
                    if (
                        needs_update is not False
                    ):  # Write if needs_update is True or None (meaning no update needed)
                        json.dump(data, temp_f)
                    temp_f.write("\n")

                except json.JSONDecodeError:
                    print(
                        f"[red]Error decoding JSON on line {processed_count}. Skipping line."
                    )
                    error_count += 1
                    # Optionally write the bad line back or log it elsewhere
                    # temp_f.write(line + "\n") # Write original line back if desired
                except Exception as e:
                    print(
                        f"[red]Unexpected error processing line {processed_count}: {e}. Skipping line."
                    )
                    error_count += 1
                    # Optionally write the bad line back
                    # temp_f.write(line + "\n")

            # End of processing loop

        # Replace the original file with the temporary file
        print("\nReplacing original file with updated version...")
        shutil.move(str(temp_file_path), str(cache_file))
        print("[green]File replacement complete.")
        temp_file_path = None  # Avoid deletion in finally block

    except Exception as e:
        print(f"[bold red]An error occurred during the update process: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up the temporary file if it still exists (e.g., due to an error before move)
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                print(
                    f"[yellow]Cleaned up temporary file due to error: {temp_file_path}"
                )
            except Exception as e_del:
                print(f"[red]Error deleting temporary file {temp_file_path}: {e_del}")

    print("\n--- Update Summary ---")
    print(f"Processed lines: {processed_count}")
    print(f"Lines updated:   {updated_count}")
    print(f"Lines skipped (errors): {error_count}")
    print("[bold green]Cache update process finished.[/bold green]")


if __name__ == "__main__":
    app()
