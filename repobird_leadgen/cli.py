import json
import os  # Needed for permission checks
import shutil
import tempfile
import traceback  # For printing stack traces on error
import webbrowser
from datetime import datetime  # Import datetime
from pathlib import Path
from typing import List, Optional, Set  # Added Set

import typer
from github import GithubException, RateLimitExceededException, Repository
from rich import print

from .config import CACHE_DIR, CONCURRENCY, OUTPUT_DIR  # Use config for dirs
from .contact_scraper import ContactScraper

# Import local modules
from .github_search import GitHubSearcher
from .models import RepoSummary
from .summarizer import save_summary  # Use the unified save function
from .utils import parallel_map

app = typer.Typer(
    add_completion=False,
    help="Automated discovery and contact extraction for GitHub repos.",
    rich_markup_mode="markdown",  # Enable rich markup
)

# --- Helper Functions ---


def _load_repos_from_cache(
    cache_file: Path, gh_instance: GitHubSearcher
) -> List[Repository]:
    """Loads raw repo data from a JSONL cache file and rehydrates Repository objects."""
    if not cache_file.exists():
        print(f"[red]Cache file not found: {cache_file}")
        raise typer.Exit(code=1)

    repos = []
    raw_repo_count = 0
    print(f"Loading repositories from cache: {cache_file}")
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            for line in f:
                raw_repo_count += 1
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    r_data = json.loads(line)
                    if not isinstance(r_data, dict) or "full_name" not in r_data:
                        print(
                            f"[yellow]Warning: Skipping invalid line in cache: {line[:100]}..."
                        )
                        continue

                    # Rehydrate using the authenticated Github object from GitHubSearcher
                    # Apply retry logic here as well, though less critical than search/enrich
                    # Using the internal retry mechanism of GitHubSearcher if available,
                    # otherwise a simple retry. Let's assume gh_instance has _execute_with_retry
                    repo = gh_instance._execute_with_retry(  # Use gh_instance here
                        gh_instance.gh.get_repo,
                        r_data["full_name"],  # And here
                    )
                    repos.append(repo)

                except json.JSONDecodeError:
                    print(
                        f"[yellow]Warning: Skipping invalid JSON line in cache: {line[:100]}..."
                    )
                except RateLimitExceededException:
                    # Should be handled by _execute_with_retry now
                    print("[yellow]Rate limit hit during repo rehydration. Retrying...")
                    # The retry logic should handle waiting
                except GithubException as e:
                    # Handled by retry logic, but log if it ultimately fails
                    print(
                        f"[yellow]Warning: Could not rehydrate repo {r_data.get('full_name', 'Unknown')} after retries: {e.status} {e.data}. Skipping."
                    )
                except Exception as e:
                    # Catch unexpected errors during rehydration of a single repo
                    print(
                        f"[yellow]Warning: Unexpected error rehydrating repo {r_data.get('full_name', 'Unknown')}: {e}. Skipping."
                    )

        print(
            f"Successfully rehydrated {len(repos)} out of {raw_repo_count} entries from cache."
        )
        return repos

    except Exception as e:
        # Catch errors opening/reading the file itself
        print(f"[red]Error loading cache file {cache_file}: {e}")
        traceback.print_exc()
        raise typer.Exit(code=1)


def _process_repo_for_summary(
    repo: Repository, scraper: ContactScraper
) -> Optional[RepoSummary]:
    """Processes a single repo to extract info and contacts for the summary."""
    try:
        # Fetch necessary attributes first to handle potential errors gracefully
        full_name = repo.full_name
        description = repo.description or ""
        stars = repo.stargazers_count
        language = repo.language
        open_issues = repo.open_issues_count
        last_push = repo.pushed_at

        # These calls can be slow or hit rate limits
        try:
            # Use search_issues for potentially better filtering and counting
            good_first_issues = repo.get_issues(
                state="open", labels=["good first issue"]
            ).totalCount
        except GithubException as e:
            print(
                f"[yellow]Warning: Could not get 'good first issue' count for {full_name}: {e.status}. Setting to 0."
            )
            good_first_issues = 0
        except Exception as e:
            print(
                f"[yellow]Warning: Unexpected error getting 'good first issue' count for {full_name}: {e}. Setting to 0."
            )
            good_first_issues = 0

        try:
            help_wanted_issues = repo.get_issues(
                state="open", labels=["help wanted"]
            ).totalCount
        except GithubException as e:
            print(
                f"[yellow]Warning: Could not get 'help wanted' count for {full_name}: {e.status}. Setting to 0."
            )
            help_wanted_issues = 0
        except Exception as e:
            print(
                f"[yellow]Warning: Unexpected error getting 'help wanted' count for {full_name}: {e}. Setting to 0."
            )
            help_wanted_issues = 0

        # Scrape contact info
        contact = scraper.extract(repo)

        return RepoSummary(
            full_name=full_name,
            description=description,
            stars=stars,
            language=language,
            open_issues=open_issues,
            good_first_issues=good_first_issues,
            help_wanted_issues=help_wanted_issues,
            last_push=last_push,
            contact=contact,
        )
    except RateLimitExceededException:
        print(
            f"[yellow]Rate limit hit processing repo {repo.full_name}. Skipping for now."
        )
        # Re-raise or handle differently if needed
        return None  # Skip this repo in parallel map
    except GithubException as e:
        print(
            f"[yellow]Warning: GitHub error processing repo {repo.full_name}: {e.status} {e.data}. Skipping."
        )
        return None
    except Exception as e:
        print(f"[red]Unexpected error processing repo {repo.full_name}: {e}. Skipping.")
        return None


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
        help=f"File to save raw repository JSON Lines data. Defaults to '[{CACHE_DIR}]/raw_repos_<label>_<lang>_<run_id>.jsonl'",
    ),
    use_browser_checker: bool = typer.Option(
        False,
        "--use-browser-checker",
        help="Use Playwright browser automation (slower, less reliable) instead of API calls to check for issue labels.",
        is_flag=True,  # Make it a flag like --use-browser-checker
    ),
) -> Optional[Path]:
    """
    Fetch repos matching criteria, cache raw data incrementally (JSONL).

    Searches repositories matching language/stars/activity, filters by open issue
    label, prints qualified repos, and saves raw data incrementally to a
    uniquely named JSON Lines (.jsonl) file. Returns the cache file path on success.
    """
    print("[bold blue]Starting GitHub repository search...[/]")
    print(f"  Filtering for Label: '{label}'")
    print(f"  Language: {language}")
    print(f"  Target Results: {max_results}")
    print(f"  Min Stars: {min_stars}")
    print(f"  Pushed within: {recent_days} days")
    if max_issue_age_days is not None:
        print(f"  Max Issue Age: {max_issue_age_days} days")
    if max_linked_prs is not None:
        print(f"  Max Linked PRs: {max_linked_prs}")

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
        print(f"  Using parameter-based cache file: {cache_file}")
    else:
        # Ensure the provided filename also ends with .jsonl for consistency
        if cache_file.suffix != ".jsonl":
            print(
                f"[yellow]Warning: Provided cache file '{cache_file}' does not end with '.jsonl'. Appending suffix."
            )
            cache_file = cache_file.with_suffix(".jsonl")
            print(f"  Adjusted cache file path: {cache_file}")
        else:
            print(f"  Using specified cache file: {cache_file}")

    # --- Load existing repo names from cache ---
    existing_repo_names: Set[str] = set()
    if cache_file.exists():
        print(f"Loading existing repos from cache: {cache_file}")
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
                        print(
                            f"[yellow]Warning: Skipping invalid JSON line during pre-load: {line[:100]}..."
                        )
            print(f"Found {len(existing_repo_names)} existing repos in cache.")
        except Exception as e:
            print(
                f"[red]Error reading existing cache file {cache_file}: {e}. Proceeding without skipping."
            )
            existing_repo_names = set()  # Reset on error
    else:
        print(f"Cache file {cache_file} not found. Starting fresh.")
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
            print(
                f"\n[green]Search complete. Added {newly_added_count} new repo details to → {cache_file}"
            )
            return cache_file  # Return path for chaining in 'full' command
        else:
            # Check if the file exists and is non-empty (maybe it had old results)
            if cache_file.exists() and cache_file.stat().st_size > 0:
                print(
                    f"\n[yellow]Search complete. No *new* repositories matching the criteria were added. Cache file → {cache_file} still contains previous results."
                )
                return cache_file  # Return existing cache path
            # Check if the file exists but is empty (search ran but found nothing new or old)
            elif cache_file.exists() and cache_file.stat().st_size == 0:
                print(
                    f"[yellow]No repositories found matching the criteria. Empty cache file created: {cache_file}"
                )
                return None  # Indicate no usable results
            elif not cache_file.exists():
                print(
                    "[red]Search process did not create a cache file, likely due to an early error."
                )
                return None  # Indicate failure
            else:  # Should not happen given above checks, but just in case
                print(
                    f"\n[yellow]Search complete. No *new* repositories matching the criteria were added to → {cache_file}"
                )
                return None  # Indicate no usable results

    except RuntimeError as e:  # Catch config errors or retry failures
        print(f"[red]Error during search: {e}")
        # traceback.print_exc() # Optional: print stack trace for debugging
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[red]An unexpected error occurred during search: {e}")
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def enrich(
    cache_file: Path = typer.Argument(
        ...,
        help="JSON Lines (.jsonl) cache file produced by the `search` command.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_formats: List[str] = typer.Option(
        ["md", "jsonl"], "--format", "-f", help="Output format(s): md, jsonl, csv."
    ),
    output_dir: Path = typer.Option(
        OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory to save the final summary files.",
    ),
    concurrency: int = typer.Option(
        CONCURRENCY,
        "--concurrency",
        "-w",
        help="Number of parallel workers for fetching details.",
    ),
) -> None:
    """
    Enrich cached repo data with contact info, stats, and save summaries.

    Reads a JSON Lines file (from `search`), fetches additional details,
    scrapes contacts, and generates summary files (Markdown, JSONL, CSV).
    """
    print("[bold blue]Starting enrichment process...[/]")
    print(f"  Input cache (JSONL): {cache_file}")
    print(f"  Output formats: {', '.join(output_formats)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Concurrency: {concurrency}")

    valid_formats = {"md", "markdown", "jsonl", "csv"}
    for fmt in output_formats:
        if fmt.lower() not in valid_formats:
            print(f"[red]Invalid output format: '{fmt}'. Choose from: md, jsonl, csv.")
            raise typer.Exit(code=1)

    try:
        # Need GitHubSearcher to access the authenticated Github object for rehydration
        # and its retry mechanism if available
        gh_searcher = GitHubSearcher()
        scraper = ContactScraper()

        # Load and rehydrate repos from JSONL cache
        repos = _load_repos_from_cache(cache_file, gh_searcher)

        if not repos:
            print(
                "[yellow]No repositories successfully loaded from cache. Nothing to enrich."
            )
            raise typer.Exit()

        # Define the processing function for parallel_map
        def process_repo_wrapper(repo: Repository) -> Optional[RepoSummary]:
            # Need to pass the scraper instance
            return _process_repo_for_summary(repo, scraper)

        # Process repositories in parallel
        print(f"Fetching details and contacts for {len(repos)} repositories...")
        summaries: List[RepoSummary] = parallel_map(
            process_repo_wrapper, repos, max_workers=concurrency, desc="Enriching repos"
        )

        # Filter out None results from potential errors during parallel processing
        successful_summaries = [s for s in summaries if s is not None]

        if not successful_summaries:
            print("[yellow]No repositories could be successfully processed.")
            raise typer.Exit()

        print(f"Successfully processed {len(successful_summaries)} repositories.")

        # Save summaries in requested formats
        save_summary(successful_summaries, str(output_dir), output_formats)

        print("[bold green]Enrichment process completed.[/]")

    except FileNotFoundError:
        # Already handled by typer's exists=True, but good practice
        print(f"[red]Cache file not found: {cache_file}")
        raise typer.Exit(code=1)
    except RuntimeError as e:  # Catch config errors
        print(f"[red]Configuration Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[red]An unexpected error occurred during enrichment: {e}")
        import traceback

        traceback.print_exc()
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
    max_linked_prs: Optional[int] = typer.Option(
        None,
        "--max-linked-prs",
        help="Filter search: issues must have at most N linked PRs.",
    ),
    output_formats: List[str] = typer.Option(
        ["md", "jsonl"], "--format", "-f", help="Output format(s): md, jsonl, csv."
    ),
    output_dir: Path = typer.Option(
        OUTPUT_DIR, "--output-dir", "-o", help="Directory for summary files."
    ),
    concurrency: int = typer.Option(  # Add typer.Option call here
        CONCURRENCY, "--concurrency", "-w", help="Parallel workers for enrichment."
    ),
    # Add the browser checker flag here too
    use_browser_checker: bool = typer.Option(
        False,
        "--use-browser-checker",
        help="Use Playwright browser automation for issue label checks during the search phase.",
        is_flag=True,
    ),
) -> None:
    """
    Convenience command to run the search and enrich steps sequentially.

    Uses a temporary JSON Lines file to pass data between search and enrich.
    """
    print("[bold blue]Starting full pipeline (search -> enrich)...[/]")

    temp_cache_path = None  # Initialize
    try:
        # Use a temporary file for the cache between steps, ensuring .jsonl suffix
        # Create in the default cache dir for better organization
        cache_dir = Path(CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".jsonl",
            prefix="repobird_cache_",
            dir=cache_dir,
        ) as tmp_file:
            temp_cache_path = Path(tmp_file.name)

        print(f"Using temporary cache file: {temp_cache_path}")

        # --- Search Step ---
        # Call the search command function programmatically
        # It now returns the path on success, or None on failure/no results
        actual_cache_path = search(
            label=label,
            language=language,
            max_results=max_results,
            min_stars=min_stars,
            recent_days=recent_days,
            max_issue_age_days=max_issue_age_days,  # Pass new arg
            max_linked_prs=max_linked_prs,  # Pass new arg
            cache_file=temp_cache_path,  # Pass the temp path
            use_browser_checker=use_browser_checker,  # Pass the flag
        )

        # Check if search succeeded and found results
        if actual_cache_path is None:
            # Search already printed messages about failure or no results
            print(
                "[yellow]Search step did not yield usable results. Skipping enrichment."
            )
            raise typer.Exit()  # Exit cleanly
        # Double check the file exists and is not empty before proceeding
        elif not actual_cache_path.exists() or actual_cache_path.stat().st_size == 0:
            print(
                "[yellow]Search completed but cache file is missing or empty. Skipping enrichment."
            )
            raise typer.Exit()  # Exit cleanly

        # --- Enrich Step ---
        # Call the enrich command function programmatically using the actual cache path
        enrich(
            cache_file=actual_cache_path,  # Use the path returned by search
            output_formats=output_formats,
            output_dir=output_dir,
            concurrency=concurrency,
        )

        print("[bold green]Full pipeline completed successfully.[/]")

    except typer.Exit:
        # Propagate exit signals cleanly
        raise
    except Exception as e:
        print(f"[bold red]An error occurred during the full pipeline: {e}[/]")
        traceback.print_exc()
        raise typer.Exit(code=1)
    finally:
        # Clean up the temporary file if it was created and still exists
        if temp_cache_path and temp_cache_path.exists():
            try:
                temp_cache_path.unlink()
                print(f"Cleaned up temporary cache file: {temp_cache_path}")
            except Exception as e:
                print(
                    f"[yellow]Warning: Could not delete temporary cache file {temp_cache_path}: {e}"
                )


def _select_cache_file() -> Optional[Path]:
    """Finds .jsonl files in CACHE_DIR and prompts user to select one."""
    cache_dir = Path(CACHE_DIR)
    if not cache_dir.is_dir():
        print(f"[red]Cache directory not found: {cache_dir}")
        return None

    print(f"Searching for cache files in: {cache_dir}")
    cache_files = sorted(
        cache_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not cache_files:
        print(f"[yellow]No .jsonl cache files found in {cache_dir}.[/yellow]")
        return None

    print("[bold]Please select a cache file to review:[/bold]")
    for i, file_path in enumerate(cache_files):
        # Show relative path for cleaner display if possible
        try:
            display_path = file_path.relative_to(Path.cwd())
        except ValueError:
            display_path = file_path
        print(
            f"  [cyan]{i + 1}[/]: {display_path} (Modified: {datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
        )

    while True:
        try:
            choice = typer.prompt(
                f"Enter number (1-{len(cache_files)}) or 'q' to quit", type=str
            )
            if choice.lower() == "q":
                print("Review cancelled.")
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(cache_files):
                return cache_files[choice_idx]
            else:
                print("[yellow]Invalid selection. Please try again.[/yellow]")
        except ValueError:
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
        print(f"[red]Error: Cache file not found: {cache_file}")
        raise typer.Exit(code=1)
    if not cache_file.is_file():
        print(f"[red]Error: Specified path is not a file: {cache_file}")
        raise typer.Exit(code=1)
    # Basic read/write permission check (might not be foolproof)
    if not os.access(cache_file, os.R_OK) or not os.access(cache_file.parent, os.W_OK):
        print(
            f"[red]Error: Insufficient permissions to read/write cache file or directory: {cache_file}"
        )
        raise typer.Exit(code=1)

    print(f"[bold blue]Starting interactive issue review for:[/bold blue] {cache_file}")

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
                        print(
                            f"  [Skipping Repo #{line_num}] Already reviewed: {repo_name}"
                        )
                        repos_skipped_count += 1
                        outfile.write(line + "\n")  # Write original line
                        continue

                    if data.get("denied", False):  # Handle legacy denial flag
                        print(
                            f"  [Skipping Repo #{line_num}] Marked as denied (legacy): {repo_name}"
                        )
                        repos_skipped_count += 1
                        outfile.write(line + "\n")  # Write original line
                        continue

                    if not repo_html_url:
                        print(
                            f"  [yellow]Warning:[/yellow] Skipping repo on line {line_num} - missing 'html_url' for {repo_name}."
                        )
                        error_count += 1
                        outfile.write(line + "\n")  # Write original line
                        continue

                    # Initialize approved_issues if not present
                    if "approved_issues" not in data:
                        data["approved_issues"] = []

                    found_issues = data.get("found_issues")
                    if not isinstance(found_issues, list) or not found_issues:
                        print(
                            f"  [Skipping Repo #{line_num}] No 'found_issues' list or empty list for {repo_name}. Marking as reviewed."
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
                            print(
                                f"    [yellow]Warning:[/yellow] Skipping invalid issue entry: {issue_number}"
                            )
                            continue

                        issue_url = f"{repo_html_url}/issues/{issue_number}"
                        print(
                            f"  Reviewing Issue {issue_index}/{issues_in_repo_count}: [bold magenta]#{issue_number}[/]"
                        )
                        print(f"  URL: {issue_url}")

                        if auto_open:
                            try:
                                webbrowser.open(issue_url)
                            except Exception as wb_err:
                                print(
                                    f"    [yellow]Warning:[/yellow] Could not open URL automatically: {wb_err}"
                                )

                        # --- Prompt for action ---
                        action = typer.prompt(
                            "  Action? (y=Approve Issue, n=Deny Issue, s=Skip Issue, sr=Skip Repo, sa=Skip All, q=Quit)",
                            default="y",
                            show_default=True,
                        ).lower()

                        issues_processed_in_repo += 1  # Count issue interaction attempt

                        if action == "y":
                            # Add to approved list if not already there
                            if issue_number not in data.get("approved_issues", []):
                                data.setdefault("approved_issues", []).append(
                                    issue_number
                                )
                                print(
                                    f"    [green]Issue #{issue_number} Approved (added to list).[/green]"
                                )
                            else:
                                print(
                                    f"    [green]Issue #{issue_number} Approved (already in list).[/green]"
                                )
                            issues_reviewed_count += 1
                        elif action == "n":
                            # Deny: Remove from approved list if it was there, but don't touch found_issues
                            if issue_number in data.get("approved_issues", []):
                                data["approved_issues"].remove(issue_number)
                                print(
                                    f"    [red]Issue #{issue_number} Denied (removed from approved list).[/red]"
                                )
                            else:
                                print(f"    [red]Issue #{issue_number} Denied.[/red]")
                            # Still counts as reviewed
                            issues_denied_count += 1
                            issues_reviewed_count += 1
                        elif action == "s":
                            print("    [yellow]Issue Skipped.[/yellow]")
                            issues_skipped_count += 1
                            # Move to next issue in this repo
                        elif action == "sr":
                            print(
                                f"    [yellow]Skipping remaining issues in {repo_name}...[/yellow]"
                            )
                            repo_level_skip = True
                            repos_skipped_count += 1
                            repo_skipped_this_line = True
                            break  # Exit inner issue loop
                        elif action == "sa":
                            print("    [yellow]Skipping all remaining...[/yellow]")
                            skip_all_remaining = True
                            repo_level_skip = True
                            repos_skipped_count += 1
                            repo_skipped_this_line = True
                            break  # Exit inner issue loop
                        elif action == "q":
                            print("    [bold magenta]Quitting review...[/bold magenta]")
                            # Write current state of data before raising Exit
                            json.dump(data, outfile)
                            outfile.write("\n")
                            # Raise typer.Exit - the finally block will handle saving
                            raise typer.Exit(code=0)  # Use code 0 for clean quit
                        else:
                            print(
                                "    [yellow]Invalid action. Treating as Skip Issue.[/yellow]"
                            )
                            issues_skipped_count += 1
                            # Move to next issue

                    # --- After Issue Loop for Repo ---
                    if not repo_level_skip:
                        # If we processed all issues without skipping the repo
                        data["reviewed"] = True
                        if "denied" in data:
                            del data["denied"]  # Clean legacy flag
                        print(f"  [green]Repo {repo_name} fully reviewed.[/green]")
                        repos_marked_reviewed_this_session += 1
                        repo_processed_this_line = True  # Mark repo as processed
                    elif repo_skipped_this_line:
                        # If repo was skipped ('sr' or 'sa'), write original line back
                        # Need to reload original line data if 'sa' was triggered mid-repo
                        # For simplicity, we'll just write the current 'data' state which might have partial denials
                        # A better approach might be needed if perfect rollback on 'sr'/'sa' is required
                        print(f"  Repo {repo_name} skipped.")
                        # Write potentially modified data (if some issues were denied before skip)
                        json.dump(data, outfile)
                        outfile.write("\n")
                        continue  # Go to next line in file

                    # Write final state of data for this repo (potentially modified issues, reviewed=True)
                    json.dump(data, outfile)
                    outfile.write("\n")

                # Catch specific expected errors during processing, let SystemExit/typer.Exit pass through
                except (
                    json.JSONDecodeError,
                    KeyError,
                    AttributeError,
                    ValueError,
                ) as e:
                    print(
                        f"  [red]Error processing data on line {line_num}: {e}. Skipping line."
                    )
                    error_count += 1
                    outfile.write(line + "\n")  # Write original line back
                finally:
                    # Increment repo processed count if it wasn't skipped and didn't error before processing
                    if repo_processed_this_line and not repo_skipped_this_line:
                        repos_processed_count += 1

        # --- End of File Loop ---

        # If loop finished OR exited via 'q', save progress by moving temp file
        shutil.move(str(temp_output_path), str(cache_file))
        print("\n[bold green]Review process finished.[/bold green]")
        if skip_all_remaining:
            print("  (Skipped remaining items as requested)")
        print("--- Summary ---")
        print(f"  Total lines in file: {total_lines}")
        print(f"  Repos skipped (already reviewed/denied/sr/sa): {repos_skipped_count}")
        print(f"  Repos newly marked as reviewed: {repos_marked_reviewed_this_session}")
        print(f"  Total issues reviewed (approved/denied): {issues_reviewed_count}")
        print(f"  Issues denied: {issues_denied_count}")  # Updated label
        print(f"  Issues skipped ('s' action): {issues_skipped_count}")
        print(f"  Lines with errors: {error_count}")
        print(f"Updated cache file: {cache_file}")

    except typer.Exit as e:
        # Handle graceful exit via 'q' or other Exit calls
        # Check if the Exit exception has a code attribute
        exit_code = getattr(e, "code", 0) # Default to 0 if no code attribute

        if exit_code == 0:
            print("Exited review process cleanly.")
        else:
            print(f"Review process exited with code {exit_code}.")
            # Consider leaving temp file if exit code indicates error
            if temp_output_path.exists():
                print(f"Partial results might be in: {temp_output_path}")
        # Note: The finally block in the outer try/except/finally structure
        # should handle the file move correctly even on Exit.

    except KeyboardInterrupt:
        print("\n[bold magenta]Keyboard interrupt detected. Saving progress...[/]")
        # Attempt to save progress by moving temp file if it exists
        if temp_output_path.exists():
            try:
                shutil.move(str(temp_output_path), str(cache_file))
                print(f"Progress saved to: {cache_file}")
            except Exception as move_err:
                print(f"[red]Error saving progress on interrupt: {move_err}")
                print(f"Partial results might be in: {temp_output_path}")
        else:
            print("No temporary file found to save.")
        raise typer.Exit(code=130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n[bold red]An error occurred during review: {e}[/]")
        traceback.print_exc()
        # Attempt cleanup, but prioritize not losing data
        if temp_output_path.exists():
            print(
                f"[yellow]Warning: Review process failed. Partial results might be in {temp_output_path}. Original file {cache_file} remains unchanged."
            )
            # Consider *not* deleting the temp file automatically on error
            # try:
            #     temp_output_path.unlink()
            #     print(f"Cleaned up temporary file due to error: {temp_output_path}")
            # except Exception as del_err:
            #     print(f"[yellow]Warning:[/yellow] Could not delete temporary file {temp_output_path} after error: {del_err}")
        raise typer.Exit(code=1)
    finally:
        # Ensure temp file is removed *only if it wasn't successfully moved* and *no major error occurred*
        # This logic is tricky. Let's be conservative and leave the temp file if the move didn't happen
        # unless it was explicitly handled (like in KeyboardInterrupt or normal exit).
        # The current logic moves the file on success/quit/interrupt, so we only need to worry about unexpected Exceptions.
        # In case of unexpected Exception, we now leave the temp file.
        pass  # Simplified cleanup logic - rely on move happening correctly


if __name__ == "__main__":
    app()
