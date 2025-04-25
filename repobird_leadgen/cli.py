import json
import tempfile
from pathlib import Path
from typing import List, Optional
import typer
from rich import print
from github import GithubException, RateLimitExceededException, Repository
import time

# Import local modules after potential environment setup (e.g., dotenv in config)
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
    """Loads raw repo data from cache and rehydrates Repository objects."""
    if not cache_file.exists():
        print(f"[red]Cache file not found: {cache_file}")
        raise typer.Exit(code=1)
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            print(
                f"[red]Invalid cache file format: Expected a JSON list in {cache_file}"
            )
            raise typer.Exit(code=1)

        print(f"Rehydrating {len(raw_data)} repositories from cache: {cache_file}")
        # Rehydrate Repository objects - this might hit rate limits if done naively
        # Consider just passing raw data if full object isn't needed immediately
        repos = []
        for r_data in raw_data:
            try:
                # Use get_repo for potentially better caching/handling by PyGithub
                repo = gh_instance.gh.get_repo(r_data["full_name"])
                repos.append(repo)
            except RateLimitExceededException:
                print("[yellow]Rate limit hit during repo rehydration. Waiting 60s...")
                time.sleep(60)
                repo = gh_instance.gh.get_repo(r_data["full_name"])  # Retry once
                repos.append(repo)
            except GithubException as e:
                print(
                    f"[yellow]Warning: Could not rehydrate repo {r_data.get('full_name', 'Unknown')}: {e.status} {e.data}. Skipping."
                )
            except Exception as e:
                print(
                    f"[yellow]Warning: Unexpected error rehydrating repo {r_data.get('full_name', 'Unknown')}: {e}. Skipping."
                )

        return repos
    except json.JSONDecodeError:
        print(f"[red]Error decoding JSON from cache file: {cache_file}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[red]Error loading cache file {cache_file}: {e}")
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
        help=f"File to save raw repository JSON data. Defaults to '[{CACHE_DIR}]/raw_repos_<label>_<lang>.json'",
    ),
) -> Path:
    """
    Fetch repos matching criteria from GitHub and cache their raw JSON data.

    Saves results to a JSON file for later use with the `enrich` command.
    """
    print("[bold blue]Starting GitHub repository search...[/]")
    print(f"  Label: '{label}'")
    print(f"  Language: {language}")
    print(f"  Max Results: {max_results}")
    print(f"  Min Stars: {min_stars}")
    print(f"  Pushed within: {recent_days} days")

    # Determine default cache file path if not provided
    if cache_file is None:
        cache_dir = Path(CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(
            c if c.isalnum() else "_" for c in label
        )  # Sanitize label for filename
        safe_lang = "".join(c if c.isalnum() else "_" for c in language)
        cache_file = cache_dir / f"raw_repos_{safe_label}_{safe_lang}.json"

    print(f"  Cache file: {cache_file}")

    try:
        gh = GitHubSearcher()  # Uses GITHUB_TOKEN from config/.env
        repos = gh.search(
            label=label,
            language=language,
            max_results=max_results,
            min_stars=min_stars,
            recent_days=recent_days,
        )
    except RuntimeError as e:  # Catch config errors (e.g., missing token)
        print(f"[red]Configuration Error: {e}")
        raise typer.Exit(code=1)
    except RateLimitExceededException:
        print(
            "[red]GitHub rate limit exceeded during search. Please wait and try again, or use a token with higher limits."
        )
        raise typer.Exit(code=1)
    except GithubException as e:
        print(f"[red]GitHub API Error during search: {e.status} {e.data}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[red]An unexpected error occurred during search: {e}")
        raise typer.Exit(code=1)

    if not repos:
        print("[yellow]No repositories found matching the criteria.")
        # Still write an empty list to the cache file for consistency
        data = []
    else:
        # Extract raw data for caching
        data = [r.raw_data for r in repos]

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[green]Saved {len(data)} raw repo details to cache → {cache_file}")
    except Exception as e:
        print(f"[red]Error writing cache file {cache_file}: {e}")
        raise typer.Exit(code=1)

    return cache_file  # Return path for chaining in 'full' command


@app.command()
def enrich(
    cache_file: Path = typer.Argument(
        ...,
        help="JSON cache file produced by the `search` command.",
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

    Reads a JSON file (from `search`), fetches additional details,
    scrapes contacts, and generates summary files (Markdown, JSONL, CSV).
    """
    print("[bold blue]Starting enrichment process...[/]")
    print(f"  Input cache: {cache_file}")
    print(f"  Output formats: {', '.join(output_formats)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Concurrency: {concurrency}")

    valid_formats = {"md", "markdown", "jsonl", "csv"}
    for fmt in output_formats:
        if fmt.lower() not in valid_formats:
            print(f"[red]Invalid output format: '{fmt}'. Choose from: md, jsonl, csv.")
            raise typer.Exit(code=1)

    try:
        # Need GitHubSearcher mainly to access the authenticated Github object
        gh_searcher = GitHubSearcher()
        scraper = ContactScraper()

        # Load and rehydrate repos from cache
        repos = _load_repos_from_cache(cache_file, gh_searcher)

        if not repos:
            print("[yellow]No repositories loaded from cache. Nothing to enrich.")
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

        traceback.print_exc()  # Print stack trace for debugging
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
    concurrency: int = typer.Option(
        CONCURRENCY, "--concurrency", "-w", help="Parallel workers for enrichment."
    ),
) -> None:
    """
    Convenience command to run the search and enrich steps sequentially.

    Uses a temporary file to pass data between search and enrich.
    """
    print("[bold blue]Starting full pipeline (search -> enrich)...[/]")

    # Use a temporary file for the cache between steps
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json", prefix="repobird_cache_"
    ) as tmp_file:
        temp_cache_path = Path(tmp_file.name)

    print(f"Using temporary cache file: {temp_cache_path}")

    try:
        # --- Search Step ---
        # Call the search command function programmatically
        cache_path = search(
            label=label,
            language=language,
            max_results=max_results,
            min_stars=min_stars,
            recent_days=recent_days,
            cache_file=temp_cache_path,  # Pass the temp path
        )

        # Check if search produced a valid cache file (it should always return the path)
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            # Check if it's just an empty list (valid case)
            is_empty_list = False
            try:
                with cache_path.open("r") as f:
                    content = f.read().strip()
                    if content == "[]":
                        is_empty_list = True
            except Exception:
                pass  # Ignore errors reading the file here

            if not is_empty_list:
                print(
                    "[red]Search step failed to produce a valid cache file. Aborting."
                )
                raise typer.Exit(code=1)
            else:
                print("[yellow]Search found no repositories. Skipping enrichment.")
                raise typer.Exit()

        # --- Enrich Step ---
        # Call the enrich command function programmatically
        enrich(
            cache_file=cache_path,
            output_formats=output_formats,
            output_dir=output_dir,
            concurrency=concurrency,
        )

        print("[bold green]Full pipeline completed successfully.[/]")

    except typer.Exit:
        # Propagate exit signals from sub-commands
        # print("[yellow]Pipeline terminated.") # Optional message
        raise
    except Exception as e:
        print(f"[bold red]An error occurred during the full pipeline: {e}[/]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)
    finally:
        # Clean up the temporary file
        if temp_cache_path.exists():
            try:
                temp_cache_path.unlink()
                print(f"Cleaned up temporary cache file: {temp_cache_path}")
            except Exception as e:
                print(
                    f"[yellow]Warning: Could not delete temporary cache file {temp_cache_path}: {e}"
                )


if __name__ == "__main__":
    app()
