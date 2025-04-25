from __future__ import annotations
from typing import Iterator, Optional, Dict, Tuple, Set, List  # Added Set, List
import random  # For jitter
from github import Github, Auth
from github.Repository import Repository
from tqdm import tqdm
from .config import GITHUB_TOKEN, CACHE_DIR  # Added CACHE_DIR
from datetime import datetime, timedelta, timezone
import time
import json  # For saving raw data incrementally
from pathlib import Path  # For cache file path handling
from github import RateLimitExceededException, GithubException, UnknownObjectException

# Import the new browser checker
from .browser_checker import BrowserIssueChecker


# Base query for repositories, label is handled separately
_REPO_QUERY_TEMPLATE = (
    "archived:false fork:false stars:>={min_stars}"
    " language:{language} pushed:>{pushed_after}"
)


class GitHubSearcher:
    """
    Searches GitHub for repositories using the API and optionally checks for
    issue labels using either the API or a browser-based checker.
    """

    def __init__(self, token: str | None = None, use_browser_checker: bool = False):
        """
        Initializes the searcher.

        Args:
            token: GitHub PAT. Defaults to GITHUB_TOKEN env var.
            use_browser_checker: If True, use Playwright to check issue labels via frontend.
                                 If False (default), use the API.
        """
        if not (token or GITHUB_TOKEN):
            raise RuntimeError(
                "GITHUB_TOKEN not found in environment or passed explicitly."
            )

        auth = Auth.Token(token or GITHUB_TOKEN)
        self.gh = Github(auth=auth, per_page=100, retry=5, timeout=15)
        self.use_browser_checker = use_browser_checker
        # Initialize checker later, only if needed, within a context manager
        self._browser_checker_instance: Optional[BrowserIssueChecker] = None

        # --- Issue Label Cache ---
        self.issue_cache_path = Path(CACHE_DIR) / "issue_label_cache.jsonl"
        self.issue_cache: Dict[
            Tuple[str, str], Tuple[bool, List[int]]  # Store flag and issue numbers
        ] = {}  # Key: (repo_full_name, label), Value: (has_label, [issue_numbers])
        self._load_issue_cache()
        # --- End Issue Label Cache ---

        print(
            f"[GitHubSearcher] Initialized. Using Browser Checker: {self.use_browser_checker}. Issue cache loaded with {len(self.issue_cache)} entries."
        )

    def _load_issue_cache(self):
        """Loads the issue label check results from the cache file."""
        if not self.issue_cache_path.exists():
            print(
                f"[GitHubSearcher] Issue cache file not found: {self.issue_cache_path}. Starting fresh."
            )
            return

        print(f"[GitHubSearcher] Loading issue cache from: {self.issue_cache_path}")
        count = 0
        try:
            with self.issue_cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Basic validation - could add timestamp check later
                        # Load issue numbers, default to empty list if missing
                        if "repo" in data and "label" in data and "has_label" in data:
                            issue_numbers = data.get(
                                "issue_numbers", []
                            )  # Default to []
                            self.issue_cache[(data["repo"], data["label"])] = (
                                data["has_label"],
                                issue_numbers,
                            )
                            count += 1
                        else:
                            print(
                                f"[yellow]Warning: Skipping invalid line in issue cache: {line[:100]}..."
                            )
                    except json.JSONDecodeError:
                        print(
                            f"[yellow]Warning: Skipping invalid JSON line in issue cache: {line[:100]}..."
                        )
            print(f"[GitHubSearcher] Loaded {count} entries into issue cache.")
        except Exception as e:
            print(f"[red]Error loading issue cache file {self.issue_cache_path}: {e}")
            # Decide whether to proceed with an empty cache or raise error
            self.issue_cache = {}  # Start fresh if loading fails

    def _append_to_issue_cache(
        self,
        repo_full_name: str,
        label: str,
        has_label: bool,
        issue_numbers: List[int],
        html_url: str,
    ):
        """Appends a new result (including html_url and issue numbers) to the issue label cache file."""
        try:
            # Ensure cache directory exists
            self.issue_cache_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "repo": repo_full_name,
                "label": label,
                "has_label": has_label,
                "issue_numbers": issue_numbers,  # Add the list of numbers
                "html_url": html_url,  # Add the URL
                "checked_at": datetime.now(timezone.utc).isoformat(),  # Add timestamp
            }
            with self.issue_cache_path.open("a", encoding="utf-8") as f:
                json.dump(entry, f)
                f.write("\n")
        except Exception as e:
            print(
                f"[red]Error appending to issue cache file {self.issue_cache_path}: {e}"
            )

    def _build_repo_query(
        self, *, language: str, min_stars: int, recent_days: int
    ) -> str:
        """Builds the query for the initial repository search (without label)."""
        pushed_after_date = datetime.now(timezone.utc) - timedelta(days=recent_days)
        pushed_after_str = pushed_after_date.strftime("%Y-%m-%d")
        return _REPO_QUERY_TEMPLATE.format(
            language=language,
            min_stars=min_stars,
            pushed_after=pushed_after_str,
        )

    def _has_open_issue_with_label(self, repo: Repository, label: str) -> bool:
        """Checks if a repository has at least one open issue with the given label."""
        # Quote the label if it contains spaces
        safe_label = f'"{label}"' if " " in label else label
        issue_query = f"repo:{repo.full_name} is:issue is:open label:{safe_label}"
        try:
            # Search for just one issue to confirm existence
            issues = self.gh.search_issues(query=issue_query)
            return issues.totalCount > 0
        except RateLimitExceededException:
            print(f"Rate limit hit checking issues for {repo.full_name}. Waiting...")
            self._wait_for_rate_limit_reset("core")  # Check 'core' limit for issues
            # Retry after waiting
            try:
                issues = self.gh.search_issues(query=issue_query)
                return issues.totalCount > 0
            except RateLimitExceededException:
                print(
                    f"Rate limit hit again after waiting for {repo.full_name}. Skipping repo."
                )
                return False
            except GithubException as ge_retry:
                print(
                    f"GitHub error checking issues for {repo.full_name} after retry: {ge_retry}. Skipping."
                )
                return False
        except GithubException as ge:
            # Handle potential 422 errors if the query is complex or repo is weird
            print(f"GitHub error checking issues for {repo.full_name}: {ge}. Skipping.")
            return False
        except Exception as e:
            print(
                f"Unexpected error checking issues for {repo.full_name}: {e}. Skipping."
            )
            return False

    def _wait_for_rate_limit_reset(
        self,
        limit_type: str = "search",
        exception: Optional[RateLimitExceededException] = None,
    ):
        """
        Waits until the GitHub API rate limit is reset, prioritizing Retry-After header.
        """
        wait_seconds = 60  # Default wait if everything else fails

        # 1. Check Retry-After header (often present for secondary limits)
        if exception and "Retry-After" in exception.headers:
            try:
                wait_seconds = int(exception.headers["Retry-After"]) + 5  # Add buffer
                print(f"Using Retry-After header: waiting {wait_seconds:.0f} seconds.")
                time.sleep(wait_seconds)
                return
            except (ValueError, TypeError):
                print("Could not parse Retry-After header.")

        # 2. Check primary rate limit reset time
        try:
            limits = self.gh.get_rate_limit()
            if limit_type == "search":
                limit_data = limits.search
            elif limit_type == "core":
                limit_data = limits.core  # Issues search uses core limit
            else:
                print(f"Unknown rate limit type '{limit_type}'. Checking search limit.")
                limit_data = limits.search

            reset_time_utc = limit_data.reset.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            calculated_wait = (
                reset_time_utc - now_utc
            ).total_seconds() + 5  # Add buffer

            if calculated_wait > 0:
                wait_seconds = calculated_wait
                print(
                    f"Waiting {wait_seconds:.0f} seconds until primary '{limit_type}' rate limit reset ({limit_data.remaining}/{limit_data.limit})."
                )
                time.sleep(wait_seconds)
                return
            else:
                print(
                    f"Primary '{limit_type}' rate limit should be reset. Proceeding cautiously."
                )
                # Even if reset, add a small delay for safety, especially if Retry-After wasn't present
                time.sleep(5)
                return

        except Exception as e:
            print(f"Error getting rate limit details: {e}. Falling back to 60s wait.")
            time.sleep(wait_seconds)  # Use the fallback wait

    def _execute_with_retry(self, func, *args, **kwargs):
        """Executes a function with exponential backoff for specific GitHub errors."""
        max_retries = 5
        base_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitExceededException as rle:
                limit_type = "search" if "search" in str(func) else "core"
                print(
                    f"\nRate limit hit (attempt {attempt + 1}/{max_retries}). Waiting..."
                )
                self._wait_for_rate_limit_reset(limit_type, exception=rle)
                # Continue to next attempt after waiting
            except (
                GithubException
            ) as ge:  # Catch other potentially transient errors like 5xx
                # Check if it's a server error (5xx) which might be transient
                if hasattr(ge, "status") and ge.status >= 500:
                    delay = (base_delay**attempt) + random.uniform(
                        0, 1
                    )  # Exponential backoff with jitter
                    print(
                        f"\nGitHub API server error (Status {ge.status}, attempt {attempt + 1}/{max_retries}): {ge.data.get('message', 'No message')}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable GitHub error (e.g., 404, 403)
                    print(f"\nNon-retryable GitHub error encountered: {ge}")
                    raise  # Re-raise the original exception
            except Exception as e:
                # Catch-all for other unexpected errors during the API call
                print(
                    f"\nUnexpected error during API call (attempt {attempt + 1}/{max_retries}): {e}"
                )
                # Depending on the error, you might want to retry or raise immediately
                # For now, let's retry for unexpected errors too, but log clearly
                if attempt < max_retries - 1:
                    delay = (base_delay**attempt) + random.uniform(0, 1)
                    print(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    print("Max retries reached for unexpected error.")
                    raise  # Re-raise after max retries

        print(
            f"\nMax retries ({max_retries}) exceeded for function {func.__name__}. Aborting operation."
        )
        # Raise a specific error or return None/empty to indicate failure
        raise RuntimeError(
            f"Failed to execute {func.__name__} after {max_retries} retries due to persistent API errors."
        )

    def search(
        self,
        *,
        label: str,  # Label is now used for filtering, not initial search
        language: str = "python",
        min_stars: int = 20,
        recent_days: int = 365,
        max_results: int = 50,  # Target number of *newly qualified* repos to find
        cache_file: Path,  # Pass cache file path for incremental writing
        existing_repo_names: Optional[Set[str]] = None,  # Added set of existing names
    ) -> Iterator[Repository]:
        """
        Searches repositories, filters by open issue label, skipping already cached repos,
        yields newly qualified repos, and saves them incrementally to the cache file.
        and saves them incrementally to the cache file (JSONL format).
        Handles rate limits with backoff and retry.
        """
        repo_query = self._build_repo_query(
            language=language, min_stars=min_stars, recent_days=recent_days
        )
        print(f"Searching GitHub repositories with query: {repo_query}")
        print(f"Filtering for repos with open issues labeled: '{label}'")
        print(f"Saving results incrementally to: {cache_file}")
        if self.use_browser_checker:
            print("[yellow]Using Browser Checker for issue labels (will be slower).[/]")

        found_count = 0
        processed_repo_count = 0
        skipped_cached_count = 0  # Initialize counter for skipped cached repos

        # Ensure cache directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine which checker function to use
        # We need to handle the browser checker's setup/teardown lifecycle
        checker_context = BrowserIssueChecker() if self.use_browser_checker else None

        try:
            # Start browser checker context if needed
            if checker_context:
                checker_context.__enter__()  # Manually enter context

            # Initial search for repositories - wrap in retry logic
            paginated_list = self._execute_with_retry(
                self.gh.search_repositories,
                query=repo_query,
                sort="updated",
                order="desc",
            )

            if (
                not paginated_list
            ):  # Handle case where initial search fails after retries
                print("[red]Initial repository search failed after multiple retries.")
                return  # Yield nothing

            print(
                f"Found {paginated_list.totalCount} potential repos matching base criteria."
            )
            print(
                f"Checking them for open '{label}' issues until {max_results} are found..."
            )

            repo_iterator = iter(paginated_list)

            # Open cache file for appending JSON Lines
            with (
                cache_file.open("a", encoding="utf-8") as f_cache,
                tqdm(
                    total=max_results, desc=f"Finding repos w/ '{label}' issues"
                ) as pbar,
            ):
                while found_count < max_results:
                    try:
                        # Get next repo from iterator with retry for iterator errors
                        repo = self._execute_with_retry(next, repo_iterator)
                        processed_repo_count += 1
                        repo_full_name = repo.full_name  # Get once

                        # --- Skip if already in the main cache file ---
                        if (
                            existing_repo_names
                            and repo_full_name in existing_repo_names
                        ):
                            # print(f"  Skipping {repo_full_name} (already in main cache: {cache_file.name})")
                            pbar.set_postfix_str(
                                "Skipping cached", refresh=False
                            )  # Update progress bar status
                            skipped_cached_count += 1  # Increment the counter
                            continue  # Skip to the next repo from GitHub search

                        # --- Check Issue Label Cache (secondary cache) ---
                        cache_key = (repo_full_name, label)
                        if cache_key in self.issue_cache:
                            has_label, issue_numbers_cache = self.issue_cache[
                                cache_key
                            ]  # Unpack tuple
                            print(
                                f"  [Cache Hit] Repo: {repo_full_name}, Label: '{label}', Has Label: {has_label}, Issues: {issue_numbers_cache}"
                            )
                            # Skip API/Browser check if cache hit
                            pbar.set_postfix_str(
                                f"Checking {repo_full_name}...", refresh=True
                            )  # Update progress bar status
                        else:
                            # --- Cache Miss - Choose Checking Method ---
                            # print(f"  [Cache Miss] Checking Repo: {repo_full_name}, Label: '{label}'...") # Can be verbose
                            pbar.set_postfix_str(
                                f"Checking {repo_full_name}...", refresh=True
                            )  # Update progress bar status
                            # Result will be a tuple: (bool, List[int]) or None if check fails
                            check_result: Optional[Tuple[bool, List[int]]] = None
                            issue_numbers_check: List[
                                int
                            ] = []  # Store numbers from check

                            if self.use_browser_checker and checker_context:
                                # Use Browser Checker
                                try:
                                    # Browser checker now returns a tuple
                                    check_result = (
                                        checker_context.check_repo_for_issue_label(
                                            repo_full_name, label
                                        )
                                    )
                                except Exception as browser_err:
                                    # Catch errors from the browser check itself
                                    print(
                                        f"[red]Error during browser check for {repo_full_name}: {browser_err}. Skipping repo."
                                    )
                                    continue  # Skip to next repo
                            else:
                                # Use API Checker (with retry)
                                try:
                                    # API check only returns boolean
                                    api_has_label_result = self._execute_with_retry(
                                        self._has_open_issue_with_label, repo, label
                                    )
                                    # Simulate the tuple structure for consistency
                                    check_result = (
                                        api_has_label_result,
                                        [],
                                    )  # API doesn't give numbers
                                except RuntimeError as api_retry_err:
                                    # Catch failure after retries from API check
                                    print(
                                        f"[red]API check failed for {repo_full_name} after retries: {api_retry_err}. Skipping repo."
                                    )
                                    continue  # Skip to next repo
                                except GithubException as ghe:
                                    print(
                                        f"[red]GitHub API error during label check for {repo_full_name}: {ghe}. Skipping repo."
                                    )
                                    continue  # Skip to next repo

                            # --- Update Cache if check was successful ---
                            if check_result is not None:
                                has_label, issue_numbers_check = (
                                    check_result  # Unpack result
                                )
                                self.issue_cache[cache_key] = (
                                    has_label,
                                    issue_numbers_check,
                                )  # Store tuple
                                self._append_to_issue_cache(
                                    repo_full_name,
                                    label,
                                    has_label,
                                    issue_numbers_check,
                                    repo.html_url,
                                )
                                print(
                                    f"  [Check Done] Repo: {repo_full_name}, Label: '{label}', Has Label: {has_label}, Issues: {issue_numbers_check}. Cached."
                                )
                            else:
                                # Check failed (API or Browser), skip this repo for qualification
                                print(
                                    f"  [Check Failed] Skipping qualification for {repo_full_name} due to check error."
                                )
                                continue  # Go to next repo in the main loop

                        # --- Process Qualification ---
                        # REMOVE the first redundant 'if has_label:' block entirely.
                        # The check below handles everything correctly based on the 'has_label'
                        # value determined from the cache or the check result.

                        # This is the correct place to check the final 'has_label' value
                        # (whether from cache or from a successful check)
                        if has_label:
                            found_count += 1  # Increment count only ONCE here
                            pbar.update(1)
                            pbar.set_postfix_str(
                                "Found qualified", refresh=False
                            )  # Update progress bar status
                            # Print details immediately
                            print(f"\n  [+] Qualified: {repo.full_name}")
                            print(f"      URL: {repo.html_url}")
                            print(f"      ({found_count}/{max_results})")

                            # Determine which issue numbers list to use (cache or check result)
                            numbers_to_save = []
                            if cache_key in self.issue_cache:
                                _, numbers_to_save = self.issue_cache[
                                    cache_key
                                ]  # Use cached numbers
                            elif check_result is not None:
                                _, numbers_to_save = (
                                    check_result  # Use numbers from the check
                                )

                            # Add the found issue numbers to the raw data before saving
                            # Use a custom key to avoid conflicts with GitHub API fields
                            repo.raw_data["_repobird_found_issue_numbers"] = (
                                numbers_to_save
                            )

                            # Write raw data (now including issue numbers) to cache file immediately
                            json.dump(repo.raw_data, f_cache)
                            f_cache.write("\n")  # Newline for JSONL format
                            f_cache.flush()  # Ensure it's written to disk

                            yield repo  # Yield the qualified repo object

                    except StopIteration:
                        print("\nReached end of GitHub repository search results.")
                        break  # Exit the while loop cleanly
                    except (RuntimeError, GithubException, UnknownObjectException) as e:
                        # Catch errors from retry logic or specific GitHub issues
                        print(f"\n[yellow]Warning: Skipping repo due to error: {e}")
                        # Decide if the error is fatal for the whole search
                        if isinstance(e, GithubException) and e.status == 422:
                            print(
                                "[red]Stopping search due to persistent invalid query or data error (422)."
                            )
                            break
                        continue  # Continue to the next repo if possible
                    except Exception as e:
                        # Catch unexpected errors during the loop processing
                        print(
                            f"\n[red]Unexpected error processing repo stream: {e}. Skipping."
                        )
                        # Consider adding a delay or stopping based on error type
                        continue

            if found_count < max_results:
                print(
                    f"\nWarning: Found only {found_count} repositories meeting all criteria after checking {processed_repo_count} candidates."
                )
            else:
                print(
                    f"\nSuccessfully found and cached {found_count} repositories meeting all criteria."
                )
            # Report skipped count if any were skipped
            if skipped_cached_count > 0:
                print(
                    f"Skipped {skipped_cached_count} repositories already present in the cache file."
                )

        except RuntimeError as e:
            # Catch failure from initial search retry wrapper
            print(f"[red]Error during initial search setup: {e}")
            # No need to return/yield anything, function ends
        except GithubException as e:
            # Catch non-retryable errors during initial search setup
            print(f"[red]GitHub API error during initial search setup: {e}")
            if e.status == 401:
                raise RuntimeError("Bad GitHub credentials. Check GITHUB_TOKEN.") from e
            # Other non-retryable errors caught here
        except Exception as e:
            # Catch any other unexpected errors during setup
            print(f"[red]An unexpected error occurred during search setup: {e}")
            raise  # Re-raise critical setup errors
        finally:
            # Ensure browser checker is closed if it was used
            if checker_context:
                checker_context.__exit__(None, None, None)  # Manually exit context

    # These helper methods are less useful now search yields results,
    # but can be kept for potential internal use or adapted if needed.
    # They would need modification to handle the yielding nature of search().
    # def good_first_issue_repos(self, **kwargs) -> Iterator[Repository]:
    #     yield from self.search(label="good first issue", **kwargs)

    # def help_wanted_repos(self, **kwargs) -> Iterator[Repository]:
    #     yield from self.search(label="help wanted", **kwargs)
