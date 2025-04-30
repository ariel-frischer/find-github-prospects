import json
import os  # Needed for permission checks
import shutil
import tempfile
import logging  # Import logging
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer

# Remove unused GitHub specific imports if not needed for rehydration anymore
# from github import GithubException, RateLimitExceededException, Repository
# from rich import print # Remove direct print import

from .config import CACHE_DIR, CONCURRENCY, OUTPUT_DIR

# Remove ContactScraper import
# from .contact_scraper import ContactScraper
# Import local modules
from .enricher import enrich_repo_entry  # Import the parallel processing helper
from .github_search import GitHubSearcher

# Import new models and summarizer function
from .logging_setup import setup_logging  # Import the setup function
from .utils import parallel_map_and_save

# Get a logger for this module
logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Automated discovery and contact extraction for GitHub repos.",
    rich_markup_mode="markdown",  # Enable rich markup
)


# --- Typer Callback for Logging Setup ---
@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    # Add options here if you want to control logging via CLI flags
    # log_level: str = typer.Option("INFO", help="Set logging level (DEBUG, INFO, WARNING, ERROR)"),
):
    """
    Main callback to setup logging before any command runs.
    """
    # Prevent running setup logic if a command is not being invoked (e.g. --help)
    if ctx.invoked_subcommand is None:
        return

    # level = getattr(logging, log_level.upper(), logging.INFO)
    level = logging.INFO # Keep it simple for now
    command_name = ctx.invoked_subcommand or "main"
    setup_logging(command_name, level)


# --- Helper Functions (Removed old ones) ---


def _load_repo_data_from_cache(cache_file: Path) -> List[Dict[str, Any]]:
    """Loads raw repo data dictionaries from a JSONL cache file."""
    if not cache_file.exists():
        logger.error(f"Cache file not found: {cache_file}")
        raise typer.Exit(code=1)

    repo_data_list = []
    logger.info(f"Loading repository data from cache: {cache_file}")
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Basic validation: check if it's a dict and has essential keys
                    if (
                        not isinstance(data, dict)
                        or "full_name" not in data
                        or "html_url" not in data
                        or "found_issues" not in data
                    ):
                        logger.warning(
                            f"Skipping invalid/incomplete line {line_num} in cache: {line[:100]}..."
                        )
                        continue
                    repo_data_list.append(data)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping invalid JSON line {line_num} in cache: {line[:100]}..."
                    )
        logger.info(
            f"Successfully loaded {len(repo_data_list)} repository entries from cache."
        )
        return repo_data_list
    except Exception as e:
        logger.exception(f"Error loading cache file {cache_file}: {e}")
        # traceback.print_exc() # logger.exception includes traceback
        raise typer.Exit(code=1)


# --- CLI Commands ---


@app.command()
def search(
    label: str = typer.Option(
        "good first issue",
        "--label",
        "-l",
        help="Issue label to search for (e.g., 'good first issue', 'help wanted').",
    ),
    language: str = typer.Option(
        "python",
        "--language",
        "-lang",
        help="Primary programming language of the repository.",
    ),
    max_results: int = typer.Option(
        30, "--max-results", "-n", help="Maximum number of repositories to fetch."
    ),
    min_stars: int = typer.Option(
        20, "--min-stars", "-s", help="Minimum number of stars required."
    ),
    recent_days: int = typer.Option(
        365,
        "--recent-days",
        "-d",
        help="Repository must have been pushed to within this many days.",
    ),
    max_issue_age_days: Optional[int] = typer.Option(
        None,
        "--max-issue-age-days",
        help="Filter repositories to only include those with at least one open issue (matching the label) created within the last N days.",
    ),
    max_linked_prs: Optional[int] = typer.Option(
        None,
        "--max-linked-prs",
        help="Filter repositories to only include those with at least one open issue (matching the label) linked to at most N pull requests.",
    ),
    cache_file: Path = typer.Option(
        None,
        "--cache-file",
        "-c",
        help=f"File to save raw repository JSON Lines data. Defaults to '[{CACHE_DIR}]/raw_repos_... .jsonl'",
    ),
    use_browser_checker: bool = typer.Option(
        False,
        "--use-browser-checker",
        help="[Search Phase] Use Playwright browser automation (slower, less reliable) instead of API calls to check for issue labels.",
        is_flag=True,
    ),
) -> Optional[Path]:
    """
    Fetch repos matching criteria, cache raw data incrementally (JSONL).

    Searches repositories matching language/stars/activity, filters by open issue
    label, prints qualified repos, and saves raw data incrementally to a
    uniquely named JSON Lines (.jsonl) file. Returns the cache file path on success.
    """
    logger.info("Starting GitHub repository search...")
    logger.info(f"  Filtering for Label: '{label}'")
    logger.info(f"  Language: {language}")
    logger.info(f"  Target Results: {max_results}")
    logger.info(f"  Min Stars: {min_stars}")
    logger.info(f"  Pushed within: {recent_days} days")
    if max_issue_age_days is not None:
        logger.info(f"  Max Issue Age: {max_issue_age_days} days")
    if max_linked_prs is not None:
        logger.info(f"  Max Linked PRs: {max_linked_prs}")

    # Determine default cache file path if not provided
    if cache_file is None:
        cache_dir = Path(CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        safe_lang = "".join(c if c.isalnum() else "_" for c in language)
        # Generate filename based on parameters
        filename_parts = [
            "raw_repos",
            f"label_{safe_label}",
            f"lang_{safe_lang}",
            f"stars_{min_stars}",
            f"days_{recent_days}",
        ]
        # Add new filter params to filename if they are used
        if max_issue_age_days is not None:
            filename_parts.append(f"age_{max_issue_age_days}")
        if max_linked_prs is not None:
            filename_parts.append(f"prs_{max_linked_prs}")

        cache_filename = "_".join(filename_parts) + ".jsonl"
        cache_file = cache_dir / cache_filename
        logger.info(f"Using parameter-based cache file: {cache_file}")
    else:
        # Ensure the provided filename also ends with .jsonl for consistency
        if cache_file.suffix != ".jsonl":
            logger.warning(
                f"Provided cache file '{cache_file}' does not end with '.jsonl'. Appending suffix."
            )
            cache_file = cache_file.with_suffix(".jsonl")
            logger.info(f"Adjusted cache file path: {cache_file}")
        else:
            logger.info(f"Using specified cache file: {cache_file}")

    # --- Load existing repo names from cache ---
    existing_repo_names: Set[str] = set()
    if cache_file.exists():
        logger.info(f"Loading existing repos from cache: {cache_file}")
        try:
            with cache_file.open("r", encoding="utf-8") as f_cache_read:
                for line in f_cache_read:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Use .get() for safer access to 'full_name'
                        repo_name = data.get("full_name")
                        if isinstance(data, dict) and repo_name:
                            existing_repo_names.add(repo_name)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping invalid JSON line during pre-load: {line[:100]}..."
                        )
            logger.info(f"Found {len(existing_repo_names)} existing repos in cache.")
        except Exception as e:
            logger.error(
                f"Error reading existing cache file {cache_file}: {e}. Proceeding without skipping."
            )
            existing_repo_names = set()  # Reset on error
    else:
        logger.info(f"Cache file {cache_file} not found. Starting fresh.")
    # --- End loading existing repo names ---

    newly_added_count = 0  # Changed from repo_count
    try:
        # Pass the flag to the constructor
        gh = GitHubSearcher(use_browser_checker=use_browser_checker)

        # search now yields results and handles incremental writing internally
        # We just need to iterate through the results to drive the process
        for _repo in gh.search(
            label=label,
            language=language,
            max_results=max_results,
            min_stars=min_stars,
            recent_days=recent_days,
            max_issue_age_days=max_issue_age_days,  # Pass new arg
            max_linked_prs=max_linked_prs,  # Pass new arg
            cache_file=cache_file,  # Pass the path for incremental writes
            existing_repo_names=existing_repo_names,  # Pass existing names
        ):
            newly_added_count += 1  # Count only newly added repos
            # Printing is now handled inside gh.search after finding qualified repo

        # Final status message based on whether any *new* repos were found/cached
        if newly_added_count > 0:
            logger.info(
                f"Search complete. Added {newly_added_count} new repo details to {cache_file}"
            )
            return cache_file  # Return path for chaining in 'full' command
        else:
            # Check if the file exists and is non-empty (maybe it had old results)
            if cache_file.exists() and cache_file.stat().st_size > 0:
                logger.warning(
                    f"Search complete. No *new* repositories matching the criteria were added. Cache file {cache_file} still contains previous results."
                )
                return cache_file  # Return existing cache path
            # Check if the file exists but is empty (search ran but found nothing new or old)
            elif cache_file.exists() and cache_file.stat().st_size == 0:
                logger.warning(
                    f"No repositories found matching the criteria. Empty cache file created: {cache_file}"
                )
                return None  # Indicate no usable results
            elif not cache_file.exists():
                logger.error(
                    "Search process did not create a cache file, likely due to an early error."
                )
                return None  # Indicate failure
            else:  # Should not happen given above checks, but just in case
                logger.warning(
                    f"Search complete. No *new* repositories matching the criteria were added to {cache_file}"
                )
                return None  # Indicate no usable results

    except RuntimeError as e:  # Catch config errors or retry failures
        logger.error(f"Error during search: {e}")
        # traceback.print_exc() # Optional: print stack trace for debugging
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during search: {e}")
        # traceback.print_exc() # logger.exception includes traceback
        raise typer.Exit(code=1)


@app.command()
def enrich(
    cache_file: Path = typer.Argument(
        ...,
        help="JSON Lines (.jsonl) cache file produced by the `search` command containing repo data.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory to save the enriched data file.",
        writable=True,  # Ensure we can write here
    ),
    concurrency: int = typer.Option(
        CONCURRENCY,
        "--concurrency",
        "-w",
        help="Number of parallel workers for scraping and analyzing issues.",
    ),
    # Add option to skip Playwright install check? Maybe not needed.
    # Add option for LLM model? Use env var for now.
) -> Optional[Path]:
    """
    Enrich cached repo data by scraping and analyzing GitHub issues using Playwright and LLM.

    Reads a JSON Lines file (from `search`), scrapes content for each issue listed
    in `found_issues`, analyzes it using an LLM (via LiteLLM), and saves the
    original repo data along with the analysis results to a new JSON Lines file
    in the specified output directory.
    """
    logger.info("Starting enrichment process (Issue Scraping & Analysis)...")
    logger.info(f"  Input cache (JSONL): {cache_file}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Concurrency: {concurrency}")
    logger.info(
        f"  LLM Model: {os.getenv('LLM_MODEL', 'gemini/gemini-1.5-pro-preview-0409')}"
    )  # Show model being used

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file path
    output_filename = f"enriched_{cache_file.stem}.jsonl"
    output_file = output_dir / output_filename

    try:
        # Load raw repo data dictionaries from the cache file
        repo_data_list = _load_repo_data_from_cache(cache_file)

        if not repo_data_list:
            logger.warning(
                "No valid repository data found in cache file. Nothing to enrich."
            )
            raise typer.Exit()

        # Process repositories in parallel, saving incrementally
        logger.info(f"Starting parallel enrichment for {len(repo_data_list)} repositories...")
        parallel_map_and_save(
            fn=enrich_repo_entry,
            items=repo_data_list,
            output_file=output_file,
            max_workers=concurrency,
            desc="Enriching Repos (Scraping & Analyzing Issues)",  # Desc for potential future overall progress bar
        )

        # Since saving is incremental, we don't collect results here.
        # The summary needs to be simpler or derived differently if needed.
        # For now, just confirm completion.
        logger.info("Enrichment Summary:")
        logger.info(f"  Processing attempted for {len(repo_data_list)} repositories.")
        logger.info(f"  Results written incrementally to: {output_file}")
        # We could enhance the writer process to count errors/successes if a detailed summary is critical.

        logger.info(
            f"Enrichment process completed. Results saved to {output_file}"
        )
        return output_file  # Return path for chaining

    except FileNotFoundError:
        # Should be caught by _load_repo_data_from_cache or typer
        logger.error(f"Input cache file not found: {cache_file}")
        raise typer.Exit(code=1)
    except (
        RuntimeError
    ) as e:  # Catch config errors (e.g., missing API keys detected by LiteLLM)
        logger.error(f"Runtime Error during enrichment: {e}")
        logger.warning(
            "Hint: Ensure necessary API keys (e.g., OPENROUTER_API_KEY) are set in your environment."
        )
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during enrichment: {e}")
        # traceback.print_exc() # logger.exception includes traceback
        raise typer.Exit(code=1)


@app.command(name="full", help="Run the full pipeline: search -> enrich.")
def full_pipeline(
    # Re-declare common options needed for both steps
    label: str = typer.Option(
        "good first issue", "--label", "-l", help="Issue label to search for."
    ),
    language: str = typer.Option(
        "python", "--language", "-lang", help="Primary programming language."
    ),
    max_results: int = typer.Option(
        30, "--max-results", "-n", help="Max repositories to fetch."
    ),
    min_stars: int = typer.Option(20, "--min-stars", "-s", help="Min stars required."),
    recent_days: int = typer.Option(
        365, "--recent-days", "-d", help="Pushed within days."
    ),
    # Add new filter options here
    max_issue_age_days: Optional[int] = typer.Option(
        None,
        "--max-issue-age-days",
        help="Filter search: issues must be created within N days.",
    ),
    max_linked_prs: Optional[int] = typer.Option(  # Add the parameter definition here
        None,
        "--max-linked-prs",
        help="[Search Phase] Filter search: issues must have at most N linked PRs.",
    ),
    # Remove output_formats, enrichment output is always JSONL now
    # output_formats: List[str] = typer.Option(
    #     ["md", "jsonl"], "--format", "-f", help="Output format(s): md, jsonl, csv."
    # ),
    output_dir: Path = typer.Option(
        OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="[Enrich Phase] Directory for the final enriched JSONL file.",
    ),
    concurrency: int = typer.Option(
        CONCURRENCY,
        "--concurrency",
        "-w",
        help="[Enrich Phase] Parallel workers for issue scraping/analysis.",
    ),
    use_browser_checker: bool = typer.Option(
        False,
        "--use-browser-checker",
        help="[Search Phase] Use Playwright browser automation for issue label checks.",
        is_flag=True,
    ),
    keep_cache: bool = typer.Option(
        False,
        "--keep-cache",
        help="Keep the intermediate raw repository cache file generated by the search step.",
        is_flag=True,
    ),
) -> None:
    """
    Run the full pipeline: search -> enrich (issue analysis).

    Finds repositories, then scrapes and analyzes their issues using Playwright/LLM.
    Uses a temporary file for the intermediate search results unless --keep-cache is specified.
    Outputs a final JSON Lines file with enriched data including issue analysis.
    """
    logger.info("Starting full pipeline (search -> enrich)...")

    temp_cache_path = None
    final_output_path = None
    try:
        # --- Determine Cache Path ---
        if keep_cache:
            # Generate a non-temporary cache filename based on search params
            cache_dir = Path(CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_label = "".join(c if c.isalnum() else "_" for c in label)
            safe_lang = "".join(c if c.isalnum() else "_" for c in language)
            filename_parts = [
                "raw_repos",
                f"label_{safe_label}",
                f"lang_{safe_lang}",
                f"stars_{min_stars}",
                f"days_{recent_days}",
            ]
            if max_issue_age_days is not None:
                filename_parts.append(f"age_{max_issue_age_days}")
            if max_linked_prs is not None:
                filename_parts.append(f"prs_{max_linked_prs}")
            cache_filename = "_".join(filename_parts) + ".jsonl"
            search_cache_path = cache_dir / cache_filename
            logger.info(f"Using persistent cache file: {search_cache_path}")
        else:
            # Use a temporary file for the cache between steps
            cache_dir = Path(CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Create the temp file manually to get its path object
            fd, temp_path_str = tempfile.mkstemp(
                suffix=".jsonl", prefix="repobird_cache_", dir=cache_dir
            )
            os.close(fd)  # Close the file descriptor, we just needed the name
            temp_cache_path = Path(temp_path_str)  # Store the path for later cleanup
            search_cache_path = temp_cache_path
            logger.info(f"Using temporary cache file: {search_cache_path}")

        # --- Search Step ---
        actual_cache_path = search(
            label=label,
            language=language,
            max_results=max_results,
            min_stars=min_stars,
            recent_days=recent_days,
            max_issue_age_days=max_issue_age_days,
            max_linked_prs=max_linked_prs,
            cache_file=search_cache_path,  # Use the determined path
            use_browser_checker=use_browser_checker,
        )

        # Check if search succeeded and found results
        if actual_cache_path is None:
            logger.warning(
                "Search step did not yield usable results. Skipping enrichment."
            )
            # No need to raise Exit here, just return, finally will clean up temp file
            return
        elif not actual_cache_path.exists() or actual_cache_path.stat().st_size == 0:
            logger.warning(
                "Search completed but cache file is missing or empty. Skipping enrichment."
            )
            # No need to raise Exit here, just return, finally will clean up temp file
            return

        # --- Enrich Step ---
        # Call the enrich command function programmatically
        # It now returns the path to the final enriched output file
        final_output_path = enrich(
            cache_file=actual_cache_path,  # Use the path from search step
            output_dir=output_dir,
            concurrency=concurrency,
        )

        if final_output_path:
            logger.info(
                f"Full pipeline completed successfully. Final output: {final_output_path}"
            )
        else:
            logger.warning(
                "Full pipeline finished, but enrichment step did not produce an output file."
            )

    except typer.Exit:
        # Propagate exit signals cleanly if raised by search() or enrich()
        raise
    except Exception as e:
        logger.exception(f"An error occurred during the full pipeline: {e}")
        # traceback.print_exc() # logger.exception includes traceback
        raise typer.Exit(code=1)
    finally:
        # Clean up the temporary cache file if it was created and still exists
        if temp_cache_path and temp_cache_path.exists():
            try:
                temp_cache_path.unlink()
                logger.info(f"Cleaned up temporary cache file: {temp_cache_path}")
            except Exception as e:
                logger.warning(
                    f"Could not delete temporary cache file {temp_cache_path}: {e}"
                )


def _select_cache_file() -> Optional[Path]:
    """Finds .jsonl files in CACHE_DIR and prompts user to select one."""
    cache_dir = Path(CACHE_DIR)
    if not cache_dir.is_dir():
        logger.error(f"Cache directory not found: {cache_dir}")
        return None

    logger.info(f"Searching for cache files in: {cache_dir}")
    cache_files = sorted(
        cache_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not cache_files:
        logger.warning(f"No .jsonl cache files found in {cache_dir}.")
        return None

    # Use Rich print for interactive parts
    print("[bold]Please select a cache file to review:[/bold]")
    for i, file_path in enumerate(cache_files):
        # Show relative path for cleaner display if possible
        try:
            display_path = file_path.relative_to(Path.cwd())
        except ValueError:
            display_path = file_path
        # Use Rich print for interactive parts
        print(
            f"  [cyan]{i + 1}[/]: {display_path} (Modified: {datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
        )

    while True:
        try:
            choice = typer.prompt(
                f"Enter number (1-{len(cache_files)}) or 'q' to quit", type=str
            )
            if choice.lower() == "q":
                logger.info("Review cancelled.")
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(cache_files):
                return cache_files[choice_idx]
            else:
                # Use Rich print for interactive parts
                print("[yellow]Invalid selection. Please try again.[/yellow]")
        except ValueError:
            # Use Rich print for interactive parts
            print("[yellow]Invalid input. Please enter a number or 'q'.[/yellow]")


@app.command()
def review(
    cache_file: Optional[Path] = typer.Argument(
        None,  # Make argument optional
        help="Path to the raw_repos_*.jsonl cache file to review. If omitted, you will be prompted to select one.",
        # Remove exists=True, readable=True, writable=True - check after selection/provision
        file_okay=True,
        dir_okay=False,
    ),
    auto_open: bool = typer.Option(
        False,
        "--open",
        "-o",
        help="Automatically open repository URLs in the browser.",
        is_flag=True,
    ),
) -> None:
    """
    Interactively review individual issues within repositories listed in a cache file.

    If CACHE_FILE is not provided, it lists .jsonl files in the cache directory
    and prompts for selection.

    Opens each issue URL (optionally automatically) and prompts for approval.
    Approved issues are added to the 'approved_issues' list. Repositories
    where all issues have been reviewed are marked with '"reviewed": true'.
    """
    # --- Select Cache File Interactively if None Provided ---
    if cache_file is None:
        cache_file = _select_cache_file()
        if cache_file is None:
            raise typer.Exit()  # Exit if no file selected or found

    # --- Validate selected/provided cache file ---
    if not cache_file.exists():
        logger.error(f"Error: Cache file not found: {cache_file}")
        raise typer.Exit(code=1)
    if not cache_file.is_file():
        logger.error(f"Error: Specified path is not a file: {cache_file}")
        raise typer.Exit(code=1)
    # Basic read/write permission check (might not be foolproof)
    if not os.access(cache_file, os.R_OK) or not os.access(cache_file.parent, os.W_OK):
        logger.error(
            f"Error: Insufficient permissions to read/write cache file or directory: {cache_file}"
        )
        raise typer.Exit(code=1)

    logger.info(f"Starting interactive issue review for: {cache_file}")

    repos_processed_count = 0
    issues_reviewed_count = 0
    issues_denied_count = 0
    repos_skipped_count = 0  # Includes already reviewed/denied, sr, sa, s actions
    issues_skipped_count = 0  # Specifically 's' action on an issue
    error_count = 0
    total_lines = 0
    skip_all_remaining = False  # Flag to skip all subsequent processing

    # Use a temporary file for writing changes
    temp_output_path = cache_file.with_suffix(cache_file.suffix + ".tmp")

    # Keep track of repos fully reviewed in this session
    repos_marked_reviewed_this_session = 0

    try:
        with (
            cache_file.open("r", encoding="utf-8") as infile,
            temp_output_path.open("w", encoding="utf-8") as outfile,
        ):
            for line_num, line in enumerate(infile, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    outfile.write("\n")  # Preserve empty lines
                    continue

                repo_processed_this_line = False
                repo_skipped_this_line = False
                try:
                    data = json.loads(line)
                    repo_name = data.get("full_name", "Unknown Repo")
                    repo_html_url = data.get("html_url")

                    # --- Skip Checks ---
                    if skip_all_remaining:
                        outfile.write(line + "\n")  # Write original line
                        repos_skipped_count += 1
                        continue

                    if data.get("reviewed", False):
                        logger.info(
                            f"Skipping Repo #{line_num} (Already reviewed): {repo_name}"
                        )
                        repos_skipped_count += 1
                        outfile.write(line + "\n")  # Write original line
                        continue

                    if data.get("denied", False):  # Handle legacy denial flag
                        logger.info(
                            f"Skipping Repo #{line_num} (Marked as denied - legacy): {repo_name}"
                        )
                        repos_skipped_count += 1
                        outfile.write(line + "\n")  # Write original line
                        continue

                    if not repo_html_url:
                        logger.warning(
                            f"Skipping repo on line {line_num} - missing 'html_url' for {repo_name}."
                        )
                        error_count += 1
                        outfile.write(line + "\n")  # Write original line
                        continue

                    # Initialize approved_issues if not present
                    if "approved_issues" not in data:
                        data["approved_issues"] = []

                    found_issues = data.get("found_issues")
                    if not isinstance(found_issues, list) or not found_issues:
                        logger.info(
                            f"Skipping Repo #{line_num} (No 'found_issues' list or empty list for {repo_name}). Marking as reviewed."
                        )
                        data["reviewed"] = True
                        if "denied" in data:
                            del data["denied"]  # Clean legacy flag
                        repos_marked_reviewed_this_session += 1
                        repos_skipped_count += 1  # Count as skipped for review purposes
                        # Write modified data back
                        json.dump(data, outfile)
                        outfile.write("\n")
                        continue

                    # --- Start Issue Review Loop ---
                    # Use Rich print for interactive parts
                    print("-" * 20)
                    print(f"Reviewing Repo #{line_num}: [bold cyan]{repo_name}[/]")
                    original_issues = list(
                        found_issues
                    )  # Create a copy to iterate over
                    issues_in_repo_count = len(original_issues)
                    issues_processed_in_repo = 0
                    repo_level_skip = False  # Flag to break inner loop

                    for issue_index, issue_number in enumerate(original_issues, 1):
                        if not isinstance(issue_number, int):
                            logger.warning(
                                f"Skipping invalid issue entry: {issue_number}"
                            )
                            continue

                        issue_url = f"{repo_html_url}/issues/{issue_number}"
                        # Use Rich print for interactive parts
                        print(
                            f"  Reviewing Issue {issue_index}/{issues_in_repo_count}: [bold magenta]#{issue_number}[/]"
                        )
                        print(f"  URL: {issue_url}")

                        if auto_open:
                            try:
                                webbrowser.open(issue_url)
                            except Exception as wb_err:
                                logger.warning(
                                    f"Could not open URL automatically: {wb_err}"
                                )

                        # --- Prompt for action --- Loop until a non-'o' action is chosen
                        while True:
                            action = typer.prompt(
                                "  Action? (y=Approve, n=Deny, s=Skip Issue, sr=Skip Repo, sa=Skip All, [bold cyan]o=Open URL[/], q=Quit)",
                                default="y",
                                show_default=True,
                            ).lower()

                            if action != "o":
                                break  # Exit the inner loop if action is not 'o'

                            # Handle 'o' action: open URL and re-prompt
                            logger.info(f"Opening URL: {issue_url}")
                            try:
                                webbrowser.open(issue_url)
                            except Exception as wb_err:
                                logger.warning(
                                    f"Could not open URL automatically: {wb_err}"
                                )
                            # continue is implicit as the while loop condition (action != 'o') is false

                        # --- Process the chosen action (y, n, s, sr, sa, q) ---

                        issues_processed_in_repo += 1  # Count issue interaction attempt

                        if action == "y":
                            # Add to approved list if not already there
                            if issue_number not in data.get("approved_issues", []):
                                data.setdefault("approved_issues", []).append(
                                    issue_number
                                )
                                logger.info(
                                    f"Issue #{issue_number} Approved (added to list)."
                                )
                            else:
                                logger.info(
                                    f"Issue #{issue_number} Approved (already in list)."
                                )
                            issues_reviewed_count += 1
                        elif action == "n":
                            # Deny: Remove from approved list if it was there, but don't touch found_issues
                            if issue_number in data.get("approved_issues", []):
                                data["approved_issues"].remove(issue_number)
                                logger.info(
                                    f"Issue #{issue_number} Denied (removed from approved list)."
                                )
                            else:
                                logger.info(f"Issue #{issue_number} Denied.")
                            # Still counts as reviewed
                            issues_denied_count += 1
                            issues_reviewed_count += 1
                        elif action == "s":
                            logger.info(f"Issue #{issue_number} Skipped.")
                            issues_skipped_count += 1
                            # Move to next issue in this repo
                        elif action == "sr":
                            logger.warning(
                                f"Skipping remaining issues in {repo_name}..."
                            )
                            repo_level_skip = True
                            repos_skipped_count += 1
                            repo_skipped_this_line = True
                            break  # Exit inner issue loop
                        elif action == "sa":
                            logger.warning("Skipping all remaining...")
                            skip_all_remaining = True
                            repo_level_skip = True
                            repos_skipped_count += 1
                            repo_skipped_this_line = True
                            break  # Exit inner issue loop
                        elif action == "q":
                            logger.info("Quitting review...")
                            # Write current state of data before raising Exit
                            json.dump(data, outfile)
                            outfile.write("\n")
                            # Raise typer.Exit - the finally block will handle saving
                            raise typer.Exit(code=0)  # Use code 0 for clean quit
                        else:
                            logger.warning(
                                "Invalid action. Treating as Skip Issue."
                            )
                            issues_skipped_count += 1
                            # Move to next issue

                    # --- After Issue Loop for Repo ---
                    if repo_level_skip:  # Handle sr, sa first
                        logger.info(f"Repo {repo_name} skipped.")
                        # Write potentially modified data (if some issues were denied before skip)
                        json.dump(data, outfile)
                        outfile.write("\n")
                        # No need to mark as reviewed if skipped mid-way
                        # repos_skipped_count is incremented where sr/sa is handled
                        break  # Exit the outer loop for this repo (already done by sr/sa logic)

                    # If we reach here, the inner loop completed without sr/sa
                    # Mark repo as reviewed only if we didn't skip it
                    data["reviewed"] = True
                    if "denied" in data:
                        del data["denied"]  # Clean legacy flag
                    logger.info(f"Repo {repo_name} fully reviewed.")
                    repos_marked_reviewed_this_session += 1
                    repo_processed_this_line = True  # Mark repo as processed

                    # Write final state of data for this repo (reviewed=True)
                    json.dump(data, outfile)
                    outfile.write("\n")
                    # repo_processed_this_line is already True

                # --- Inner Try/Except/Finally for processing a single line ---
                except (
                    json.JSONDecodeError,
                    KeyError,
                    AttributeError,
                    ValueError,
                ) as e:
                    logger.error(
                        f"Error processing data on line {line_num}: {e}. Skipping line."
                    )
                    error_count += 1
                    outfile.write(line + "\n")  # Write original line back
                finally:  # Belongs to the inner try (line 718)
                    # Increment repo processed count if it wasn't skipped and didn't error before processing
                    if repo_processed_this_line and not repo_skipped_this_line:
                        repos_processed_count += 1
            # --- End of File Loop (inside 'with' blocks) ---

            # --- Final operations after loop (still inside main 'try') ---
            # If loop finished OR exited via 'q', save progress by moving temp file
            # This needs to be inside the main try block to be caught by outer except/finally
            shutil.move(str(temp_output_path), str(cache_file))
            logger.info("Review process finished.")
            if skip_all_remaining:
                logger.info("  (Skipped remaining items as requested)")
            logger.info("--- Review Summary ---")
            logger.info(f"  Total lines in file: {total_lines}")
            logger.info(
                f"  Repos skipped (already reviewed/denied/sr/sa): {repos_skipped_count}"
            )
            logger.info(
                f"  Repos newly marked as reviewed: {repos_marked_reviewed_this_session}"
            )
            logger.info(f"  Total issues reviewed (approved/denied): {issues_reviewed_count}")
            logger.info(f"  Issues denied: {issues_denied_count}")  # Updated label
            logger.info(f"  Issues skipped ('s' action): {issues_skipped_count}")
            logger.info(f"  Lines with errors: {error_count}")
            logger.info(f"Updated cache file: {cache_file}")

    # --- Outer Try/Except/Finally for the whole command ---
    except typer.Exit as e:  # Correct indentation
        # Handle graceful exit via 'q' or other Exit calls
        # Check if the Exit exception has a code attribute
        exit_code = getattr(e, "code", 0)  # Default to 0 if no code attribute

        if exit_code == 0:
            logger.info("Exited review process cleanly.")
        else:
            logger.warning(f"Review process exited with code {exit_code}.")
            # Consider leaving temp file if exit code indicates error
            if temp_output_path.exists():
                logger.warning(f"Partial results might be in: {temp_output_path}")
        # Note: The finally block in the outer try/except/finally structure
        # should handle the file move correctly even on Exit.

    except KeyboardInterrupt:  # Correct indentation
        logger.warning("Keyboard interrupt detected. Saving progress...")
        # Attempt to save progress by moving temp file if it exists
        if temp_output_path.exists():
            try:
                shutil.move(str(temp_output_path), str(cache_file))
                logger.info(f"Progress saved to: {cache_file}")
            except Exception as move_err:
                logger.error(f"Error saving progress on interrupt: {move_err}")
                logger.warning(f"Partial results might be in: {temp_output_path}")
        else:
            logger.info("No temporary file found to save.")
        raise typer.Exit(code=130)  # Standard exit code for Ctrl+C
    except Exception as e:  # Correct indentation
        logger.exception(f"An error occurred during review: {e}")
        # traceback.print_exc() # logger.exception includes traceback
        # Attempt cleanup, but prioritize not losing data
        if temp_output_path.exists():
            logger.warning(
                f"Review process failed. Partial results might be in {temp_output_path}. Original file {cache_file} remains unchanged."
            )
            # Consider *not* deleting the temp file automatically on error
            # try:
            #     temp_output_path.unlink()
            #     logger.info(f"Cleaned up temporary file due to error: {temp_output_path}")
            # except Exception as del_err:
            #     logger.warning(f"Could not delete temporary file {temp_output_path} after error: {del_err}")
        raise typer.Exit(code=1)
    finally:
        # Ensure temp file is removed *only if it wasn't successfully moved* and *no major error occurred*
        # This logic is tricky. Let's be conservative and leave the temp file if the move didn't happen
        # unless it was explicitly handled (like in KeyboardInterrupt or normal exit).
        # The current logic moves the file on success/quit/interrupt, so we only need to worry about unexpected Exceptions.
        # In case of unexpected Exception, we now leave the temp file.
        pass  # Simplified cleanup logic - rely on move happening correctly


@app.command()
def post_process(
    input_file: Path = typer.Argument(
        ...,
        help="Path to the enriched JSON Lines file (output from the `enrich` command).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """
    Filters an enriched JSONL file to include only repos with 'good first issues for agent'.

    Reads the input file line by line, checks the 'issue_analysis' list in each line's JSON data.
    If any issue analysis has 'is_good_first_issue_for_agent' set to true, the entire line
    is written to a new output file named 'filtered_<original_name>.jsonl' in the same directory.
    """
    logger.info(f"Starting post-processing filter on: {input_file}")

    output_file = input_file.parent / f"filtered_{input_file.name}"
    repos_read = 0
    repos_written = 0
    errors_parsing = 0

    try:
        with (
            input_file.open("r", encoding="utf-8") as infile,
            output_file.open("w", encoding="utf-8") as outfile,
        ):
            for line in infile:
                repos_read += 1
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    data = json.loads(line)
                    issue_analysis_list = data.get("issue_analysis", [])

                    if not isinstance(issue_analysis_list, list):
                        logger.warning(
                            f"Skipping line {repos_read} - 'issue_analysis' is not a list."
                        )
                        continue

                    found_good_issue = False
                    for issue_analysis in issue_analysis_list:
                        if (
                            isinstance(issue_analysis, dict)
                            and issue_analysis.get("is_good_first_issue_for_agent")
                            is True
                        ):
                            found_good_issue = True
                            break  # Found one, no need to check further in this repo

                    # If at least one good issue was found, filter the list and write modified data
                    if found_good_issue:
                        # Create a new list containing only the good issues
                        filtered_issue_analysis = [
                            issue
                            for issue in issue_analysis_list
                            if isinstance(issue, dict)
                            and issue.get("is_good_first_issue_for_agent") is True
                        ]

                        # Modify the original data structure to only include the filtered issues
                        data["issue_analysis"] = filtered_issue_analysis

                        # Write the modified data as a JSON line
                        outfile.write(json.dumps(data) + "\n")
                        repos_written += 1

                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {repos_read} - Invalid JSON.")
                    errors_parsing += 1
                except Exception as e:
                    logger.error(f"Error processing line {repos_read}: {e}. Skipping line.")
                    errors_parsing += 1

        logger.info("--- Post-processing Summary ---")
        logger.info(f"  Total repositories read: {repos_read}")
        logger.info(f"  Repositories written to filtered file: {repos_written}")
        if errors_parsing > 0:
            logger.warning(f"  Lines skipped due to parsing errors: {errors_parsing}")
        logger.info(f"Filtered results saved to {output_file}")

    except Exception as e:
        logger.exception(f"An error occurred during post-processing: {e}")
        # traceback.print_exc() # logger.exception includes traceback
        # Clean up potentially incomplete output file on error
        if output_file.exists():
            try:
                output_file.unlink()
                logger.warning(f"Removed incomplete output file: {output_file}")
            except Exception as del_err:
                logger.error(
                    f"Error removing incomplete output file {output_file}: {del_err}"
                )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
