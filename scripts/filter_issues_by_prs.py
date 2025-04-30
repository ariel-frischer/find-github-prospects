import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import typer

# Removed unused PyGithub imports: Github, GithubException, Issue, RateLimitExceededException
from playwright.sync_api import (
    Error as PlaywrightError,
)
from playwright.sync_api import (
    Page,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from rich import print
from rich.progress import Progress

# Add project root to sys.path to allow importing repobird_leadgen modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import config after modifying sys.path (GITHUB_TOKEN import removed as it's unused)
# try:
#     from repobird_leadgen.config import GITHUB_TOKEN
# except ImportError:
#     print(
#         "[bold red]Error:[/bold red] Could not import GITHUB_TOKEN from repobird_leadgen.config."
#     )
#     print("Ensure the script is run from the project root or the package is installed.")
#     sys.exit(1)


# --- Playwright Selectors ---
# Selector for the sidebar section containing "Development"
DEV_SECTION_SELECTOR = (
    'div[data-testid="sidebar-section"]:has(h3:has-text("Development"))'
)
# Selector for the UL containing linked PRs within the Development section
LINKED_PR_LIST_SELECTOR = (
    f"{DEV_SECTION_SELECTOR} ul[data-testid='issue-viewer-linked-pr-container']"
)
# Selector for list items (actual PR links) within the container
LINKED_PR_ITEM_SELECTOR = f"{LINKED_PR_LIST_SELECTOR} li"

# --- GitHub API Interaction (Keep for potential future use or remove if fully switching) ---

# Comment out or remove GitHub instance logic if fully switching to Playwright for this script
# _github_instance: Optional[Github] = None
# def get_github_instance() -> Github: ...
# def wait_for_rate_limit_reset(...): ...


# --- GitHub API Interaction (Removed) ---
# Functions get_github_instance and wait_for_rate_limit_reset removed as they are no longer used.


# --- Playwright Check Function ---


def check_dev_section_for_prs(page: Page, issue_url: str) -> bool:
    """
    Uses Playwright to check the GitHub issue page for linked PRs in the
    'Development' sidebar section.

    Args:
        page: The Playwright Page object to use.
        issue_url: The URL of the GitHub issue.

    Returns:
        True if linked PRs are found in the Development section, False otherwise.
        Returns False on Playwright errors (e.g., timeout).
    """
    print(f"        Checking URL (Playwright): {issue_url}")
    try:
        page.goto(
            issue_url, wait_until="domcontentloaded", timeout=45000
        )  # Increased timeout

        # Wait for the Development section header to be visible (or timeout)
        # This helps ensure the sidebar has loaded before checking the list
        page.locator(f'{DEV_SECTION_SELECTOR} h3:has-text("Development")').wait_for(
            state="visible",
            timeout=20000,  # Shorter timeout for this specific wait
        )
        print("          - Development section header found.")

        # Check if the linked PR list container exists
        pr_list_container = page.locator(LINKED_PR_LIST_SELECTOR)

        if pr_list_container.count() > 0:
            # Check if the container actually has any <li> children (PR items)
            pr_items_count = pr_list_container.locator("li").count()
            print(
                f"          - Linked PR list container found. Items: {pr_items_count}"
            )
            if pr_items_count > 0:
                return True  # Found the list with PR items
            else:
                # Container exists but is empty
                return False
        else:
            # Development section exists, but no PR list container found within it
            print(
                "          - Development section found, but no linked PR list container."
            )
            return False

    except PlaywrightTimeoutError:
        print(
            f"        [yellow]Timeout waiting for elements on {issue_url}. Assuming no linked PRs.[/yellow]"
        )
        return False  # Assume no PRs if section doesn't load
    except PlaywrightError as e:
        print(
            f"        [red]Playwright error checking {issue_url}: {e}. Assuming no linked PRs.[/red]"
        )
        return False  # Assume no PRs on error
    except Exception as e:
        print(
            f"        [red]Unexpected error during Playwright check for {issue_url}: {e}[/red]"
        )
        traceback.print_exc()
        return False  # Assume no PRs on unexpected error
    finally:
        # Add a small delay to avoid overwhelming GitHub
        time.sleep(1.0)  # 1 second delay


# --- Main Application Logic ---

app = typer.Typer(
    add_completion=False,
    help="Filters an enriched JSONL file in-place, removing issues that have linked Pull Requests.",
    rich_markup_mode="markdown",
)


@app.command()
def main(
    input_file: Path = typer.Argument(
        ...,
        help="Path to the enriched JSON Lines file (e.g., output from `enrich` or `post_process`). This file will be modified in-place.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=True,  # Need write permission for in-place modification
    ),
):
    """
    Filters an enriched JSONL file in-place.

    Reads the input file line by line. For each repository, it checks every issue
    in the 'issue_analysis' list via the GitHub API to see if it has linked PRs.
    Issues *with* linked PRs are removed from the 'issue_analysis' list.
    If a repository ends up with an empty 'issue_analysis' list, the entire
    repository (line) is removed from the file.

    Uses a temporary file for safe in-place modification.
    Requires GITHUB_TOKEN environment variable (potentially, if other API calls remain).
    Requires Playwright browsers to be installed (`playwright install`).
    """
    print(
        f"[bold blue]Starting PR filter (using Playwright) for:[/bold blue] {input_file}"
    )
    print("[yellow]This script will modify the file in-place.[/yellow]")
    print("[cyan]Initializing Playwright...[/cyan]")

    # Playwright setup happens outside the file loop
    try:
        with sync_playwright() as p:
            # Consider browser choice (chromium, firefox, webkit)
            # Headless=True is generally preferred for automation
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            print("[green]Playwright initialized successfully.[/green]")

            temp_output_path = input_file.with_suffix(input_file.suffix + ".tmp")
            repos_read = 0
            repos_written = 0
            issues_checked = 0
            issues_removed = 0
            errors_parsing = 0
            errors_api = 0  # Keep for potential non-playwright errors

            try:
                with (
                    input_file.open("r", encoding="utf-8") as infile,
                    temp_output_path.open("w", encoding="utf-8") as outfile,
                    Progress() as progress,  # Add progress bar
                ):
                    # Estimate total lines for progress bar (optional but nice)
                    try:
                        total_lines = sum(1 for _ in infile)
                        infile.seek(0)  # Reset file pointer after counting
                        task = progress.add_task(
                            "[cyan]Filtering repos...", total=total_lines
                        )
                    except Exception:
                        print(
                            "[yellow]Could not determine file length for progress bar.[/yellow]"
                        )
                        task = progress.add_task(
                            "[cyan]Filtering repos..."
                        )  # Indeterminate

                    for line in infile:
                        repos_read += 1
                        line = line.strip()
                        if not line:
                            continue  # Skip empty lines but don't write them to temp

                        try:
                            data = json.loads(line)
                            # Look for 'full_name' at the top level
                            repo_full_name = data.get("full_name")
                            issue_analysis_list: List[Dict[str, Any]] = data.get(
                                "issue_analysis", []
                            )

                            # Check using repo_full_name now
                            if not repo_full_name or not isinstance(
                                issue_analysis_list, list
                            ):
                                print(
                                    f"[yellow]Warning: Skipping line {repos_read} - Missing top-level 'full_name' or invalid 'issue_analysis'. Writing original line back.[/yellow]"
                                )
                                # Write the original line back to preserve it
                                outfile.write(line + "\n")
                                errors_parsing += 1
                                progress.update(task, advance=1)
                                continue

                            if not issue_analysis_list:
                                # Repo already has no issues, skip writing it
                                print(
                                    f"  Skipping repo {repo_full_name} (no issues in list)."
                                )
                                progress.update(task, advance=1)
                                continue

                            print(
                                f"  Checking repo: {repo_full_name} ({len(issue_analysis_list)} issues)"
                            )
                            filtered_issues: List[Dict[str, Any]] = []
                            repo_api_error = (
                                False  # Keep flag for non-playwright errors
                            )

                            # REMOVED: GitHub API repo fetching logic
                            # try:
                            #     repo = g.get_repo(repo_full_name)
                            # except Exception as e: ...

                            for issue_data in issue_analysis_list:
                                issues_checked += 1
                                issue_number = issue_data.get("issue_number")
                                issue_url = issue_data.get(
                                    "issue_url"
                                )  # Get URL from data

                                if not isinstance(issue_number, int) or not issue_url:
                                    print(
                                        f"    [yellow]Skipping invalid issue entry (missing number or URL): {issue_data.get('issue_url', 'N/A')}[/yellow]"
                                    )
                                    # Keep this invalid entry? Or remove? Let's keep it for now.
                                    # filtered_issues.append(issue_data)
                                    continue

                                try:
                                    print(
                                        f"      Checking issue #{issue_number} for linked PRs via Playwright..."
                                    )
                                    # Call the new Playwright check function
                                    if not check_dev_section_for_prs(page, issue_url):
                                        print(
                                            f"        [green]Keeping issue #{issue_number} (no linked PRs found in Dev section).[/green]"
                                        )
                                        filtered_issues.append(issue_data)
                                    else:
                                        print(
                                            f"        [yellow]Removing issue #{issue_number} (linked PR found in Dev section).[/yellow]"
                                        )
                                        issues_removed += 1
                                # REMOVED: Old GitHub API exception handling for has_linked_pr
                                # except RateLimitExceededException: ...
                                except (
                                    Exception
                                ) as e:  # Catch general errors during the check loop
                                    print(
                                        f"    [red]Error processing issue #{issue_number} in {repo_full_name}: {e}. Skipping issue check, keeping issue.[/red]"
                                    )
                                    errors_api += 1  # Use general API error counter
                                    # Keep the issue if the check fails unexpectedly
                                    filtered_issues.append(issue_data)

                            # After checking all issues for the repo
                            if (
                                not repo_api_error and filtered_issues
                            ):  # repo_api_error check might be redundant now
                                # Write back only if no major API error occurred for the repo
                                # AND if there are still issues remaining after filtering
                                data["issue_analysis"] = filtered_issues
                                outfile.write(json.dumps(data) + "\n")
                                repos_written += 1
                                print(
                                    f"  Finished repo {repo_full_name}. Kept {len(filtered_issues)} issues."
                                )
                            elif not repo_api_error and not filtered_issues:
                                print(
                                    f"  Skipping repo {repo_full_name} (all issues removed or none to begin with)."
                                )
                            # If repo_api_error is True, we don't write the repo back (though this flag might be less relevant now)

                        except json.JSONDecodeError:
                            print(
                                f"[yellow]Warning: Skipping line {repos_read} - Invalid JSON.[/yellow]"
                            )
                            errors_parsing += 1
                            # Write the original line back to preserve it
                            outfile.write(line + "\n")
                        except Exception as e:
                            print(
                                f"[red]Error processing line {repos_read}: {e}. Skipping line. Writing original line back.[/red]"
                            )
                            traceback.print_exc()
                            errors_parsing += 1
                            # Write the original line back to preserve it
                            outfile.write(line + "\n")
                        finally:
                            progress.update(
                                task, advance=1
                            )  # Advance progress bar per line processed

                # After processing all lines, replace original file with temp file
                input_file.unlink()  # Remove original
                temp_output_path.rename(input_file)  # Rename temp to original name

                print("\n--- PR Filter Summary ---")
                print(f"  Repositories read: {repos_read}")
                print(f"  Repositories written (with PR-free issues): {repos_written}")
                print(f"  Total issues checked: {issues_checked}")
                print(f"  Issues removed (due to linked PRs): {issues_removed}")
                if errors_parsing > 0:
                    print(
                        f"  [yellow]Lines skipped due to parsing errors: {errors_parsing}[/yellow]"
                    )
                if errors_api > 0:
                    print(
                        f"  [red]Issues skipped due to processing errors: {errors_api}[/red]"
                    )  # Updated label
                print(f"[bold green]Filtering complete. File updated → {input_file}[/]")

            except Exception as e:
                print(f"[bold red]An error occurred during filtering: {e}[/]")
                traceback.print_exc()
                # Clean up temp file if it exists on error
                if temp_output_path.exists():
                    try:
                        temp_output_path.unlink()
                        print(
                            f"[yellow]Removed temporary file due to error: {temp_output_path}"
                        )
                    except Exception as del_err:
                        print(
                            f"[red]Error removing temporary file {temp_output_path}: {del_err}"
                        )
                sys.exit(1)
            finally:
                # Ensure browser is closed within the Playwright context
                print("[cyan]Closing Playwright browser...[/cyan]")
                browser.close()

    except Exception as e:
        print(f"[bold red]Failed to initialize or run Playwright: {e}[/bold red]")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # The main logic is now executed by the typer app when the script is run.
    # The variables previously defined here were unused because the 'main'
    # function defined above has its own scope for counters.
    app()
