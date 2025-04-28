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

# Script to update raw repo cache JSONL files.
# Handles:
# 1. Renaming legacy "_repobird_found_issues_basic" field to "found_issues".
# 2. Populating empty "found_issues" (or renamed legacy field) by searching for issue numbers with the specified label using BrowserIssueChecker.
# 3. Upgrading "found_issues" lists containing only integers (issue numbers) to lists of dictionaries with full issue details (title, description) using BrowserIssueChecker.
app = typer.Typer(
    add_completion=False,
    help="Updates/Upgrades a raw repo cache JSONL file: renames legacy fields, populates empty issue lists, and upgrades integer-only lists to full issue details using the BrowserIssueChecker.",
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
    Processes a JSONL cache file:
    - Renames `_repobird_found_issues_basic` to `found_issues`.
    - If `found_issues` is missing or empty, populates it with issue numbers found via browser check for the given label.
    - If `found_issues` contains only integers, fetches full details for those issues via browser check.
    Updates the file in place.
    """
    print(
        f"[bold blue]Updating/Upgrading cache file issue data:[/bold blue] {cache_file}"
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
                    write_back_original = (
                        False  # Flag to write original line if update fails
                    )

                    # --- 1. Handle legacy field rename ---
                    if "_repobird_found_issues_basic" in data:
                        print(
                            f"\n[cyan]Found legacy field '_repobird_found_issues_basic' in:[/cyan] {repo_name}"
                        )
                        legacy_value = data.pop(
                            "_repobird_found_issues_basic"
                        )  # Remove old key
                        # Assign to new key, even if empty, might be upgraded later
                        data["found_issues"] = legacy_value
                        needs_update = True  # Mark for potential write
                        print(
                            f"  Renamed to 'found_issues'. Current value: {legacy_value}"
                        )

                    # --- 2. Populate if 'found_issues' is missing or empty ---
                    if "found_issues" not in data or data["found_issues"] == []:
                        if "found_issues" not in data:
                            print(
                                f"\n[cyan]Field 'found_issues' missing, fetching issue numbers for:[/cyan] {repo_name}"
                            )
                        else:  # It exists but is empty
                            print(
                                f"\n[cyan]Field 'found_issues' is empty, fetching issue numbers for:[/cyan] {repo_name}"
                            )

                        try:
                            # Use check_repo_for_issue_label to find issues with the label
                            has_label, issue_details = (
                                checker.check_repo_for_issue_label(repo_name, label)
                            )

                            # Extract just the numbers
                            issue_numbers = [
                                item.get("number")
                                for item in issue_details
                                if isinstance(item, dict)
                                and item.get("number") is not None
                            ]

                            data["found_issues"] = (
                                issue_numbers  # Add/update field with numbers list
                            )
                            needs_update = True
                            print(
                                f"  [green]Populated 'found_issues' with {len(issue_numbers)} issue numbers.[/green] (Repo Has Label: {has_label})"
                            )
                            updated_count += 1
                        except Exception as check_err:
                            print(
                                f"[red]Error checking issues for {repo_name}: {check_err}. Skipping update for this line.[/red]"
                            )
                            error_count += 1
                            write_back_original = (
                                True  # Ensure original line is written on error
                            )

                    # --- 3. Upgrade if 'found_issues' contains only integers ---
                    elif isinstance(data.get("found_issues"), list) and all(
                        isinstance(item, int) for item in data["found_issues"]
                    ):
                        issue_numbers_to_fetch = data["found_issues"]
                        if not issue_numbers_to_fetch:
                            # List exists but is empty, already handled above or no action needed
                            pass
                        else:
                            print(
                                f"\n[cyan]Integer list found, fetching full details for {len(issue_numbers_to_fetch)} issues in:[/cyan] {repo_name}"
                            )
                            needs_update = True  # Mark for potential write
                            new_issue_details = []
                            temp_page = None
                            try:
                                temp_page = checker._get_new_page()
                                for issue_num in issue_numbers_to_fetch:
                                    issue_url = f"https://github.com/{repo_name}/issues/{issue_num}"
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
                                            f"    [yellow]Skipping details for issue {issue_num} due to fetch error.[/yellow]"
                                        )
                                    time.sleep(0.5)  # Small delay

                                data["found_issues"] = (
                                    new_issue_details  # Replace int list with dict list
                                )
                                print(
                                    f"  [green]Upgraded 'found_issues':[/green] Fetched details for {len(new_issue_details)} issues."
                                )
                                updated_count += 1  # Count upgrade as an update
                            except Exception as fetch_err:
                                print(
                                    f"[red]Error fetching individual issue details for {repo_name}: {fetch_err}. Skipping upgrade for this line.[/red]"
                                )
                                error_count += 1
                                write_back_original = True  # Write original line back
                            finally:
                                if temp_page:
                                    try:
                                        temp_page.close()
                                    except Exception:
                                        pass

                    # --- 4. Handle existing full details or unexpected formats ---
                    elif isinstance(data.get("found_issues"), list):
                        # Field exists, is a list, but not all integers -> assume it's already the new format (list of dicts)
                        # print(f"Skipping line {processed_count}: Already has full issue details.") # Optional verbose log
                        pass  # No update needed
                    else:
                        # Field exists but is not a list or None (unexpected format)
                        field_type = type(data.get("found_issues"))
                        print(
                            f"[yellow]Skipping line {processed_count}: Unexpected format for 'found_issues': {field_type}[/yellow]"
                        )
                        write_back_original = True  # Write original line back

                    # --- Write back ---
                    if write_back_original:
                        temp_f.write(
                            line + "\n"
                        )  # Write original line due to error or skip
                    elif needs_update:
                        json.dump(data, temp_f)  # Write updated data
                        temp_f.write("\n")
                    else:  # No update needed, write original line
                        temp_f.write(line + "\n")

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
