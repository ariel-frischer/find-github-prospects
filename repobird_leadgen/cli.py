import json
import logging  # Import logging
import multiprocessing  # Import multiprocessing
import os  # Needed for permission checks
import re  # Add re import for URL parsing
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer

# Remove unused GitHub specific imports if not needed for rehydration anymore
# from github import GithubException, RateLimitExceededException, Repository
from rich import print  # Use rich print directly
from rich.panel import Panel  # Import Panel for display

# Import BrowserIssueChecker for the new command
from .browser_checker import BrowserIssueChecker
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
    level = logging.INFO  # Keep it simple for now
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


def _select_generic_file(file_list: List[Path], prompt_message: str) -> Optional[Path]:
    """Prompts user to select a file from a list."""
    if not file_list:
        logger.warning("No files provided for selection.")
        return None

    # Use Rich print for interactive parts
    print(f"[bold]{prompt_message}[/bold]")
    for i, file_path in enumerate(file_list):
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
                f"Enter number (1-{len(file_list)}) or 'q' to quit", type=str
            )
            if choice.lower() == "q":
                logger.info("Selection cancelled.")
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(file_list):
                return file_list[choice_idx]
            else:
                # Use Rich print for interactive parts
                print("[yellow]Invalid selection. Please try again.[/yellow]")
        except ValueError:
            # Use Rich print for interactive parts
            print("[yellow]Invalid input. Please enter a number or 'q'.[/yellow]")


def _load_enriched_data(
    enriched_file: Path,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Loads enriched data into a nested dictionary for quick lookup."""
    enriched_data_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    logger.info(f"Loading enriched analysis data from: {enriched_file}")
    if not enriched_file.exists():
        logger.warning(
            f"Enriched file not found: {enriched_file}. Cannot display analysis during review."
        )
        return enriched_data_map  # Return empty map

    try:
        with enriched_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    repo_name = data.get("full_name")
                    issue_analysis_list = data.get("issue_analysis", [])

                    if not repo_name or not isinstance(issue_analysis_list, list):
                        logger.warning(
                            f"Skipping invalid line {line_num} in enriched file: Missing 'full_name' or 'issue_analysis' list."
                        )
                        continue

                    if repo_name not in enriched_data_map:
                        enriched_data_map[repo_name] = {}

                    for analysis in issue_analysis_list:
                        if isinstance(analysis, dict) and "issue_number" in analysis:
                            issue_num = analysis["issue_number"]
                            enriched_data_map[repo_name][issue_num] = analysis
                        else:
                            logger.warning(
                                f"Skipping invalid analysis entry on line {line_num} for repo {repo_name}: {analysis}"
                            )

                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping invalid JSON line {line_num} in enriched file: {line[:100]}..."
                    )
        logger.info(
            f"Successfully loaded enriched data for {len(enriched_data_map)} repositories."
        )
        return enriched_data_map
    except Exception as e:
        logger.exception(f"Error loading enriched file {enriched_file}: {e}")
        # Return empty map, review can proceed without analysis display
        return {}


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
        help="[Search Phase] Use Playwright browser automation. If detailed filters (--max-issue-age-days or --max-linked-prs) are used, this enables a hybrid approach: API for age filtering, browser for verifying '--max-linked-prs 0'. Otherwise, uses browser for initial label check.",
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
        f"  LLM Model: {os.getenv('LLM_MODEL', 'openrouter/google/gemini-2.5-flash-preview')}"
    )  # Show model being used

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file path
    output_filename = f"enriched_{cache_file.stem}.jsonl"
    output_file = output_dir / output_filename

    # --- Setup for Cost Tracking ---
    manager = multiprocessing.Manager()
    # Initialize shared dictionary with expected keys
    shared_cost_data = manager.dict(
        {
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
    )
    shared_lock = manager.Lock()
    # --- End Setup ---

    try:
        # Load raw repo data dictionaries from the cache file
        repo_data_list = _load_repo_data_from_cache(cache_file)

        if not repo_data_list:
            logger.warning(
                "No valid repository data found in cache file. Nothing to enrich."
            )
            raise typer.Exit()

        # Process repositories in parallel, saving incrementally
        logger.info(
            f"Starting parallel enrichment for {len(repo_data_list)} repositories..."
        )
        parallel_map_and_save(
            fn=enrich_repo_entry,  # The worker function
            items=repo_data_list,  # Items to process
            output_file=output_file,  # Output file path
            max_workers=concurrency,  # Max worker processes
            shared_cost_data=shared_cost_data,  # Pass shared dict
            lock=shared_lock,  # Pass shared lock
            # desc is not used by parallel_map_and_save currently
        )

        # Since saving is incremental, we don't collect results here.
        # The summary needs to be simpler or derived differently if needed.
        # For now, just confirm completion.
        logger.info("Enrichment Summary:")
        logger.info(f"  Processing attempted for {len(repo_data_list)} repositories.")
        logger.info(f"  Results written incrementally to: {output_file}")
        # We could enhance the writer process to count errors/successes if a detailed summary is critical.

        # --- Log Final Costs ---
        final_cost = shared_cost_data.get("total_cost", 0.0)
        final_input_tokens = shared_cost_data.get("total_input_tokens", 0)
        final_output_tokens = shared_cost_data.get("total_output_tokens", 0)
        logger.info("--- Enrichment Cost Summary ---")
        logger.info(f"  Total Estimated LLM Cost: ${final_cost:.6f}")
        logger.info(f"  Total Input Tokens: {final_input_tokens}")
        logger.info(f"  Total Output Tokens: {final_output_tokens}")
        # --- End Log Final Costs ---

        logger.info(f"Enrichment process completed. Results saved to {output_file}")
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
        help="[Search Phase] Use Playwright browser automation (see `search` help for details on hybrid approach).",
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


# _select_cache_file is no longer needed, replaced by _select_generic_file


@app.command()
def review(
    input_file: Optional[Path] = typer.Argument(
        None,  # Make argument optional
        help="Path to the ENRICHED file (e.g., enriched_*.jsonl or filtered_*.jsonl) to review. If omitted, you will be prompted.",
        file_okay=True,
        dir_okay=False,
        # Validation happens after selection
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
    Interactively review issues listed in an enriched file.

    If INPUT_FILE is not provided, it lists .jsonl files in the output/cache directories
    and prompts for selection.

    Displays the LLM analysis for each issue from the input file.
    Opens each issue URL (optionally automatically) and prompts for approval.
    Approved issues are added to the 'approved_issues' list in a NEW output file
    named 'reviewed_<input_filename>.jsonl'. Repositories where all issues
    have been reviewed are marked with '"reviewed": true'.
    """
    # --- Select Input File Interactively if None Provided ---
    if input_file is None:
        # Look in both output and cache dirs for potential files
        output_dir = Path(OUTPUT_DIR)
        cache_dir = Path(CACHE_DIR)
        potential_files = []
        if output_dir.is_dir():
            potential_files.extend(
                sorted(
                    output_dir.glob("*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            )
        if cache_dir.is_dir():
            # Add cache files, ensuring no duplicates if dirs overlap or are same
            for cf in sorted(
                cache_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                if cf not in potential_files:
                    potential_files.append(cf)

        if not potential_files:
            logger.error(f"No .jsonl files found in {output_dir} or {cache_dir}.")
            raise typer.Exit(code=1)

        input_file = _select_generic_file(
            potential_files, "Select enriched file to review"
        )
        if input_file is None:
            raise typer.Exit()  # Exit if no file selected

    # --- Validate selected/provided input file ---
    if not input_file.exists():
        logger.error(f"Error: Input file not found: {input_file}")
        raise typer.Exit(code=1)
    if not input_file.is_file():
        logger.error(f"Error: Specified path is not a file: {input_file}")
        raise typer.Exit(code=1)
    # Basic read/write permission check (need write for output dir)
    if not os.access(input_file, os.R_OK) or not os.access(input_file.parent, os.W_OK):
        logger.error(
            f"Error: Insufficient permissions to read input file or write to its directory: {input_file}"
        )
        raise typer.Exit(code=1)

    # Enriched data is read line-by-line, no pre-loading needed.
    logger.info(f"Starting interactive issue review for: {input_file}")

    repos_processed_count = 0
    issues_reviewed_count = 0
    issues_denied_count = 0
    repos_skipped_count = 0  # Includes already reviewed/denied, sr, sa, s actions
    issues_skipped_count = 0  # Specifically 's' action on an issue
    error_count = 0
    total_lines = 0
    skip_all_remaining = False  # Flag to skip all subsequent processing

    # --- Determine Output File Path (based on INPUT file) ---
    output_file = input_file.parent / f"reviewed_{input_file.name}"
    logger.info(f"Review results will be written to: {output_file}")
    # --- End Output File Path ---

    # --- Load already reviewed repos if output file exists ---
    reviewed_repo_names: Set[str] = set()
    output_file_exists = output_file.exists()
    if output_file_exists:
        logger.info(
            f"Output file {output_file} exists. Loading previously reviewed repos..."
        )
        try:
            with output_file.open("r", encoding="utf-8") as f_rev:
                for line in f_rev:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("reviewed") is True:
                            repo_name = data.get("full_name")
                            if repo_name:
                                reviewed_repo_names.add(repo_name)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping invalid JSON line in existing output file: {line[:100]}..."
                        )
            logger.info(
                f"Loaded {len(reviewed_repo_names)} previously reviewed repository names."
            )
        except Exception as e:
            logger.error(
                f"Error reading existing output file {output_file}: {e}. Will overwrite."
            )
            reviewed_repo_names = set()  # Reset on error, effectively overwriting
            output_file_exists = False  # Treat as if it doesn't exist for opening mode

    # Keep track of repos fully reviewed in this session
    repos_marked_reviewed_this_session = 0

    output_file_handle = None  # Keep track of the handle to close it in finally
    try:
        # Open the single INPUT enriched file for reading.
        # Open the OUTPUT file in append ('a') or write ('w') mode.
        output_mode = "a" if output_file_exists and reviewed_repo_names else "w"
        logger.info(f"Opening output file {output_file} in '{output_mode}' mode.")
        output_file_handle = output_file.open(output_mode, encoding="utf-8")
        with input_file.open("r", encoding="utf-8") as infile:
            # Use the opened output file handle directly
            outfile = output_file_handle
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

                    # --- Skip if already reviewed in existing output file ---
                    if repo_name in reviewed_repo_names:
                        logger.info(
                            f"Skipping Repo #{line_num} ({repo_name}): Already reviewed in {output_file.name}"
                        )
                        repos_skipped_count += 1
                        continue  # Skip to next line in input file

                    # --- Skip Checks (runtime flags) ---
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

                    # Initialize approved_issues if not present
                    if "approved_issues" not in data:
                        data["approved_issues"] = []

                    # Get the issue analysis list directly from the enriched data
                    issue_analysis_list = data.get("issue_analysis")
                    if (
                        not isinstance(issue_analysis_list, list)
                        or not issue_analysis_list
                    ):
                        logger.info(
                            f"Skipping Repo #{line_num} (No 'issue_analysis' list or empty list for {repo_name}). Marking as reviewed."
                        )
                        data["reviewed"] = True
                        # No need to handle 'denied' flag here as it's not part of enriched data
                        repos_marked_reviewed_this_session += 1
                        repos_skipped_count += 1  # Count as skipped for review purposes
                        # Write modified data back
                        json.dump(data, outfile)
                        outfile.write("\n")
                        continue

                    # --- Start Issue Review Loop ---
                    print("-" * 20)
                    print(f"Reviewing Repo #{line_num}: [bold cyan]{repo_name}[/]")
                    # Iterate directly over the analysis list
                    issues_in_repo_count = len(issue_analysis_list)
                    issues_processed_in_repo = 0
                    repo_level_skip = False  # Flag to break inner loop

                    for issue_index, analysis_details in enumerate(
                        issue_analysis_list, 1
                    ):
                        # Extract issue number and URL from the analysis data itself
                        issue_number = analysis_details.get("issue_number")
                        issue_url = analysis_details.get("issue_url")

                        if not isinstance(issue_number, int) or not issue_url:
                            logger.warning(
                                f"Skipping invalid analysis entry in repo {repo_name}: Missing 'issue_number' or 'issue_url'. Analysis: {analysis_details}"
                            )
                            continue  # Skip this analysis entry

                        # --- Display LLM Analysis ---
                        if analysis_details:  # Check if analysis exists for this issue
                            panel_content = ""
                            for key, value in analysis_details.items():
                                # Skip redundant fields or format nicely
                                if (
                                    key in ["issue_number", "issue_url", "error"]
                                    and value is None
                                ):
                                    continue
                                if key == "error" and value:
                                    panel_content += f"[bold red]{key.replace('_', ' ').title()}:[/] {value}\n"
                                elif isinstance(value, bool):
                                    panel_content += f"[bold]{key.replace('_', ' ').title()}:[/] {'[green]Yes[/]' if value else '[yellow]No[/]'}\n"
                                elif isinstance(value, (float, int)):
                                    panel_content += f"[bold]{key.replace('_', ' ').title()}:[/] [cyan]{value:.2f}[/]\n"
                                else:
                                    panel_content += f"[bold]{key.replace('_', ' ').title()}:[/]\n{value}\n\n"
                            # Print the panel after the loop, correctly indented
                            print(
                                Panel(
                                    panel_content.strip(),
                                    title=f"LLM Analysis for Issue #{issue_number}",
                                    border_style="blue",
                                    expand=False,
                                )
                            )
                        else:
                            # Correctly indented else block
                            print(
                                Panel(
                                    "[yellow]LLM analysis not found for this issue.[/]",
                                    title=f"Issue #{issue_number}",
                                    border_style="yellow",
                                    expand=False,
                                )
                            )
                        # --- End Display LLM Analysis ---

                        # Use Rich print for interactive parts
                        print(
                            f"  Reviewing Issue {issue_index}/{issues_in_repo_count}: [bold magenta]#{issue_number}[/]"
                        )
                        # Use logger for URL, print for interaction
                        logger.info(f"  Issue URL: {issue_url}")

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
                            logger.info(f"Opening URL in browser: {issue_url}")
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
                            logger.warning("Invalid action. Treating as Skip Issue.")
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
            # No need to move temp file, writing was direct
            logger.info("Review process finished.")
            if skip_all_remaining:
                logger.info("  (Skipped remaining items as requested)")
            logger.info("--- Review Summary ---")
            logger.info(f"  Total lines read from input file: {total_lines}")
            logger.info(
                f"  Repos skipped (previously reviewed/runtime skip): {repos_skipped_count}"
            )
            logger.info(
                f"  Repos newly reviewed in this session: {repos_marked_reviewed_this_session}"
            )
            logger.info(
                f"  Total issues reviewed (approved/denied): {issues_reviewed_count}"
            )
            logger.info(f"  Issues denied: {issues_denied_count}")  # Updated label
            logger.info(f"  Issues skipped ('s' action): {issues_skipped_count}")
            logger.info(f"  Lines with errors: {error_count}")
            logger.info(f"Reviewed results saved to: {output_file}")

    # --- Outer Try/Except/Finally for the whole command ---
    except typer.Exit as e:
        # Handle graceful exit via 'q' or other Exit calls
        exit_code = getattr(e, "code", 0)
        if exit_code == 0:
            logger.info("Exited review process cleanly.")
        else:
            logger.warning(f"Review process exited with code {exit_code}.")
        # Output file should be closed by the finally block

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected.")
        logger.info(f"Partial review results saved in: {output_file}")
        # File will be closed by finally block
        raise typer.Exit(code=130)  # Standard exit code for Ctrl+C

    except Exception as e:
        logger.exception(f"An unexpected error occurred during review: {e}")
        logger.error(
            f"Review process failed. Partial results might be in {output_file}."
        )
        # File will be closed by finally block
        raise typer.Exit(code=1)

    finally:
        # Ensure the output file handle is closed if it was opened
        if output_file_handle and not output_file_handle.closed:
            try:
                output_file_handle.close()
                logger.debug("Output file handle closed.")
            except Exception as close_err:
                logger.error(f"Error closing output file {output_file}: {close_err}")


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
                    logger.error(
                        f"Error processing line {repos_read}: {e}. Skipping line."
                    )
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


@app.command()
def test_url_search(
    repo_or_url: str = typer.Argument(
        ...,
        help="Repository name (e.g., 'owner/repo') or full GitHub issues URL to test.",
    ),
    label: str = typer.Option(
        "good first issue",
        "--label",
        "-l",
        help="Issue label to search for.",
    ),
    max_issue_age_days: Optional[int] = typer.Option(
        None,
        "--max-issue-age-days",
        help="Filter issues created within N days.",
    ),
    max_linked_prs: Optional[int] = typer.Option(
        None,
        "--max-linked-prs",
        help="Filter issues with at most N linked PRs (0 means exclude linked PRs).",
    ),
):
    """
    Test browser-based issue searching directly on a specific repository or URL.

    Uses BrowserIssueChecker to navigate to the issues page with the specified
    filters (label, age, linked PRs) and reports the found issues.
    """
    logger.info("Starting direct browser search test...")

    # Extract repo_full_name from input
    repo_full_name: Optional[str] = None
    if "github.com" in repo_or_url:
        # Try to extract from URL
        match = re.search(r"github\.com/([^/]+/[^/]+)", repo_or_url)
        if match:
            repo_full_name = match.group(1)
        else:
            logger.error(f"Could not extract 'owner/repo' from URL: {repo_or_url}")
            raise typer.Exit(code=1)
    else:
        # Assume it's already owner/repo format
        if "/" not in repo_or_url or len(repo_or_url.split("/")) != 2:
            logger.error(
                f"Invalid repository format: '{repo_or_url}'. Use 'owner/repo'."
            )
            raise typer.Exit(code=1)
        repo_full_name = repo_or_url

    logger.info(f"  Target Repository: {repo_full_name}")
    logger.info(f"  Label: '{label}'")
    if max_issue_age_days is not None:
        logger.info(f"  Max Issue Age: {max_issue_age_days} days")
    if max_linked_prs is not None:
        logger.info(f"  Max Linked PRs: {max_linked_prs}")
    else:
        logger.info("  Max Linked PRs: Not specified (no PR filter applied)")

    try:
        with BrowserIssueChecker(headless=True) as checker:  # Use context manager
            logger.info("Browser checker initialized. Performing check...")
            # Use check_repo_for_issue_label as it handles query construction
            found_flag, issue_details = checker.check_repo_for_issue_label(
                repo_full_name=repo_full_name,
                label=label,
                max_linked_prs=max_linked_prs,
                max_issue_age_days=max_issue_age_days,
            )

            logger.info("--- Browser Search Test Results ---")
            if found_flag:
                logger.info(f"Found {len(issue_details)} issues matching criteria:")
                for detail in issue_details:
                    logger.info(
                        f"  - #{detail.get('number')}: {detail.get('title', 'N/A')[:70]}..."
                    )
            else:
                logger.info("No issues found matching the specified criteria.")

    except Exception as e:
        logger.exception(f"An error occurred during the browser search test: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
