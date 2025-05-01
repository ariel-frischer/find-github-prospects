import json

# from rich import print # Remove print import
import logging  # Import logging
import sys
import time
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

# Import the moved function *after* sys.path is updated
from repobird_leadgen.browser_checker import check_dev_section_for_prs  # noqa: E402

# Import logging setup *after* sys.path is updated
from repobird_leadgen.logging_setup import setup_logging  # noqa: E402

# Get logger for this module
logger = logging.getLogger(__name__)

# Selector for the "Closed" state label within the *fixed* (non-sticky) header metadata section
CLOSED_STATE_SELECTOR = 'div[data-testid="issue-metadata-fixed"] span[data-testid="header-state"]:has-text("Closed")'


# --- GitHub API Interaction (Keep for potential future use or remove if fully switching) ---

# Comment out or remove GitHub instance logic if fully switching to Playwright for this script
# _github_instance: Optional[Github] = None
# def get_github_instance() -> Github: ...
# def wait_for_rate_limit_reset(...): ...


# --- GitHub API Interaction (Removed) ---
# Functions get_github_instance and wait_for_rate_limit_reset removed as they are no longer used.


# --- Playwright Check Function (Moved to browser_checker.py) ---


def is_issue_closed(page: Page) -> bool:
    """Checks if the issue page header shows the 'Closed' state."""
    try:
        # Check if the "Closed" state element exists and is visible
        closed_label = page.locator(CLOSED_STATE_SELECTOR)
        if closed_label.count() > 0 and closed_label.is_visible(
            timeout=5000
        ):  # Short timeout
            logger.info("Issue state is 'Closed'.")
            return True
        else:
            # logger.debug("Issue state is not 'Closed'.") # Debug
            return False
    except PlaywrightTimeoutError:
        logger.warning("Timeout checking issue state. Assuming not closed.")
        return False  # Assume open if state check times out
    except Exception as e:
        logger.warning(f"Error checking issue state: {e}. Assuming not closed.")
        return False  # Assume open on error


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
    # Setup logging for this script run
    setup_logging("filter_prs")

    logger.info(f"Starting PR filter (using Playwright) for: {input_file}")
    logger.warning("This script will modify the file in-place.")
    logger.info("Initializing Playwright...")

    # Playwright setup happens outside the file loop
    try:
        with sync_playwright() as p:
            # Consider browser choice (chromium, firefox, webkit)
            # Headless=True is generally preferred for automation
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            logger.info("Playwright initialized successfully.")

            temp_output_path = input_file.with_suffix(input_file.suffix + ".tmp")
            repos_read = 0
            repos_written = 0
            issues_checked = 0
            issues_removed_pr = 0  # Renamed counter for PRs
            issues_removed_closed = 0  # New counter for closed issues
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
                        logger.warning(
                            "Could not determine file length for progress bar."
                        )
                        task = progress.add_task(
                            "[cyan]Filtering repos...",
                            total=None,  # Indeterminate
                        )

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
                                logger.warning(
                                    f"Skipping line {repos_read} - Missing top-level 'full_name' or invalid 'issue_analysis'. Writing original line back."
                                )
                                # Write the original line back to preserve it
                                outfile.write(line + "\n")
                                errors_parsing += 1
                                progress.update(task, advance=1)
                                continue

                            if not issue_analysis_list:
                                # Repo already has no issues, skip writing it
                                logger.info(
                                    f"Skipping repo {repo_full_name} (no issues in list)."
                                )
                                progress.update(task, advance=1)
                                continue

                            logger.info(
                                f"Checking repo: {repo_full_name} ({len(issue_analysis_list)} issues)"
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
                                    logger.warning(
                                        f"Skipping invalid issue entry (missing number or URL): {issue_data.get('issue_url', 'N/A')}"
                                    )
                                    # Keep this invalid entry? Or remove? Let's keep it for now.
                                    # filtered_issues.append(issue_data)
                                    continue

                                try:
                                    # Announce which issue is being checked *before* navigation
                                    logger.info(
                                        f"Checking issue #{issue_number}: {issue_url}"
                                    )

                                    # Navigate to the issue page (needed for both checks)
                                    page.goto(
                                        issue_url,
                                        wait_until="domcontentloaded",
                                        timeout=45000,
                                    )

                                    # 1. Check if issue is closed
                                    if is_issue_closed(page):
                                        logger.info(
                                            f"Removing issue #{issue_number} (state is 'Closed')."
                                        )
                                        issues_removed_closed += (
                                            1  # Increment closed counter
                                        )
                                        time.sleep(0.5)  # Small delay after check
                                        continue  # Skip to next issue

                                    # 2. If open, check for linked PRs
                                    # No need to print issue number again here
                                    if check_dev_section_for_prs(
                                        page,
                                        issue_url,  # Pass URL for logging inside the function
                                    ):  # check_dev_section_for_prs already includes delay
                                        logger.info(
                                            f"Removing issue #{issue_number} (linked PR found in Dev section)."
                                        )
                                        issues_removed_pr += 1  # Increment PR counter
                                    else:
                                        logger.info(
                                            f"Keeping issue #{issue_number} (Open, no linked PRs found in Dev section)."
                                        )
                                        filtered_issues.append(issue_data)

                                except PlaywrightTimeoutError:
                                    logger.error(
                                        f"Timeout navigating to or checking {issue_url}. Keeping issue."
                                    )
                                    errors_api += 1
                                    filtered_issues.append(
                                        issue_data
                                    )  # Keep on timeout
                                except PlaywrightError as e:
                                    logger.error(
                                        f"Playwright error checking {issue_url}: {e}. Keeping issue."
                                    )
                                    errors_api += 1
                                    filtered_issues.append(
                                        issue_data
                                    )  # Keep on playwright error
                                except Exception as e:  # Catch other general errors during the check loop
                                    logger.exception(
                                        f"Error processing issue #{issue_number} in {repo_full_name}: {e}. Skipping issue check, keeping issue."
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
                                logger.info(
                                    f"Finished repo {repo_full_name}. Kept {len(filtered_issues)} issues."
                                )
                            elif not repo_api_error and not filtered_issues:
                                logger.info(
                                    f"Skipping repo {repo_full_name} (all issues removed or none to begin with)."
                                )
                            # If repo_api_error is True, we don't write the repo back (though this flag might be less relevant now)

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Skipping line {repos_read} - Invalid JSON."
                            )
                            errors_parsing += 1
                            # Write the original line back to preserve it
                            outfile.write(line + "\n")
                        except Exception as e:
                            logger.exception(
                                f"Error processing line {repos_read}: {e}. Skipping line. Writing original line back."
                            )
                            # traceback.print_exc() # logger.exception includes traceback
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

                logger.info("--- PR Filter Summary ---")
                logger.info(f"  Repositories read: {repos_read}")
                logger.info(
                    f"  Repositories written (with remaining issues): {repos_written}"
                )
                logger.info(f"  Total issues checked: {issues_checked}")
                logger.info(
                    f"  Issues removed (state was 'Closed'): {issues_removed_closed}"
                )
                logger.info(
                    f"  Issues removed (due to linked PRs): {issues_removed_pr}"
                )
                if errors_parsing > 0:
                    logger.warning(
                        f"  Lines skipped due to parsing errors: {errors_parsing}"
                    )
                if errors_api > 0:
                    logger.error(
                        f"  Issues skipped due to processing errors: {errors_api}"
                    )  # Updated label
                logger.info(f"Filtering complete. File updated: {input_file}")

            except Exception as e:
                logger.exception(f"An error occurred during filtering: {e}")
                # traceback.print_exc() # logger.exception includes traceback
                # Clean up temp file if it exists on error
                if temp_output_path.exists():
                    try:
                        temp_output_path.unlink()
                        logger.warning(
                            f"Removed temporary file due to error: {temp_output_path}"
                        )
                    except Exception as del_err:
                        logger.error(
                            f"Error removing temporary file {temp_output_path}: {del_err}"
                        )
                sys.exit(1)
            finally:
                # Ensure browser is closed within the Playwright context
                logger.info("Closing Playwright browser...")
                browser.close()

    except Exception as e:
        logger.exception(f"Failed to initialize or run Playwright: {e}")
        # traceback.print_exc() # logger.exception includes traceback
        sys.exit(1)


if __name__ == "__main__":
    # The main logic is now executed by the typer app when the script is run.
    # The variables previously defined here were unused because the 'main'
    # function defined above has its own scope for counters.
    app()
