#!/usr/bin/env python
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

import typer
from rich.progress import track
from rich import print

# Assume repobird_leadgen is installed or PYTHONPATH is set correctly
from repobird_leadgen.browser_checker import BrowserIssueChecker

# Script to fix our cached old jsonl items that were created before the _repobird_found_issue_numbers field was needed.
app = typer.Typer(
    add_completion=False,
    help="Updates a raw repo cache JSONL file by adding missing '_repobird_found_issue_numbers' using the BrowserIssueChecker.",
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
    Processes a JSONL cache file, finds lines missing the issue numbers field,
    runs the browser checker to get them, and updates the file in place.
    """
    print(f"[bold blue]Updating cache file:[/bold blue] {cache_file}")
    print(f"  Using label: '{label}' for checks")
    print(f"  Browser Headless: {headless}")

    updated_count = 0
    processed_count = 0
    error_count = 0

    # Create a temporary file in the same directory to handle potential move issues across filesystems
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=cache_file.parent,
            prefix=f"{cache_file.stem}_update_",
            suffix=".jsonl",
        ) as temp_f, cache_file.open("r", encoding="utf-8") as input_f, BrowserIssueChecker(
            headless=headless, check_delay=delay, timeout=timeout
        ) as checker:
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

                    # Check if issue numbers are missing
                    if "_repobird_found_issue_numbers" not in data:
                        print(
                            f"\n[cyan]Checking missing issues for:[/cyan] {repo_name}"
                        )
                        try:
                            has_label, issue_numbers = checker.check_repo_for_issue_label(
                                repo_name, label
                            )
                            # Add the field, even if the list is empty
                            data["_repobird_found_issue_numbers"] = issue_numbers
                            print(
                                f"  [green]Added issues:[/green] {issue_numbers} (Has Label: {has_label})"
                            )
                            updated_count += 1
                        except Exception as check_err:
                            print(
                                f"[red]Error checking issues for {repo_name}: {check_err}. Skipping update for this line."
                            )
                            error_count += 1
                            # Write original data back if check fails
                            temp_f.write(line + "\n")
                            continue # Skip writing updated data

                    # Write the (potentially updated) data back to the temp file
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
        temp_file_path = None # Avoid deletion in finally block

    except Exception as e:
        print(f"[bold red]An error occurred during the update process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the temporary file if it still exists (e.g., due to an error before move)
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                print(f"[yellow]Cleaned up temporary file due to error: {temp_file_path}")
            except Exception as e_del:
                print(f"[red]Error deleting temporary file {temp_file_path}: {e_del}")

    print("\n--- Update Summary ---")
    print(f"Processed lines: {processed_count}")
    print(f"Lines updated:   {updated_count}")
    print(f"Lines skipped (errors): {error_count}")
    print("[bold green]Cache update process finished.[/bold green]")


if __name__ == "__main__":
    app()
