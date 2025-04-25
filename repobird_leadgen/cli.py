import json
import tempfile
from pathlib import Path
from typing import List, Optional, Set  # Added Set
import typer
from rich import print
from github import GithubException, RateLimitExceededException, Repository
import traceback  # For printing stack traces on error

# Import local modules
from .github_search import GitHubSearcher
from .contact_scraper import ContactScraper
from .summarizer import save_summary  # Use the unified save function
from .models import RepoSummary
from .config import OUTPUT_DIR, CONCURRENCY, CACHE_DIR  # Use config for dirs
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

    # Determine default cache file path if not provided
    if cache_file is None:
        cache_dir = Path(CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        safe_lang = "".join(c if c.isalnum() else "_" for c in language)
        # Generate unique run ID using timestamp
        # Generate filename based on parameters
        filename_parts = [
            "raw_repos",
            f"label_{safe_label}",
            f"lang_{safe_lang}",
            f"stars_{min_stars}",
            f"days_{recent_days}",
        ]
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

    # We no longer delete the cache file, we append to it.
    # The search function itself handles opening in append mode.

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
            print(
                f"\n[yellow]Search complete. No *new* repositories matching the criteria were added to → {cache_file}"
            )
            # Still return the path if it exists, as enrich might still be useful
            if cache_file.exists():
                return cache_file
            # Check if the file exists but is empty (search ran but found nothing)
            if cache_file.exists() and cache_file.stat().st_size == 0:
                print(
                    f"[yellow]No repositories found matching the criteria. Empty cache file created: {cache_file}"
                )
            elif not cache_file.exists():
                print(
                    "[red]Search process did not create a cache file, likely due to an early error."
                )
            else:  # File exists but might have partial data if interrupted badly
                print(
                    f"[yellow]Search completed, but no repositories met all criteria. Cache file may contain partial data if interrupted: {cache_file}"
                )
            # Don't return cache_file if no repos were successfully found and saved.
            # This prevents 'enrich' from running on an empty/failed cache.
            return None

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
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".jsonl", prefix="repobird_cache_"
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
            cache_file=temp_cache_path,  # Pass the temp path
            use_browser_checker=use_browser_checker,  # Pass the flag
        )

        # Check if search succeeded and found results
        if actual_cache_path is None:
            # Search already printed messages about failure or no results
            print("[yellow]Search step did not yield results. Skipping enrichment.")
            raise typer.Exit()  # Exit cleanly
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
        # Clean up the temporary file if it was created
        if temp_cache_path and temp_cache_path.exists():
            try:
                temp_cache_path.unlink()
                print(f"Cleaned up temporary cache file: {temp_cache_path}")
            except Exception as e:
                print(
                    f"[yellow]Warning: Could not delete temporary cache file {temp_cache_path}: {e}"
                )


if __name__ == "__main__":
    app()
