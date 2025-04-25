from __future__ import annotations
from typing import (
    Iterator,
    Optional,
    Dict,
    Tuple,
    Set,
    List,
    Any,
)  # Added Set, List, Any
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
            Tuple[str, str],
            Tuple[bool, List[Dict[str, Any]]],  # Store flag and issue details list
        ] = {}  # Key: (repo_full_name, label), Value: (has_label, [issue_details])
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
                        # Load issue details, default to empty list if missing
                        if "repo" in data and "label" in data and "has_label" in data:
                            # Use the new key 'issue_details'
                            issue_details = data.get(
                                "issue_details", []
                            )  # Default to []
                            self.issue_cache[(data["repo"], data["label"])] = (
                                data["has_label"],
                                issue_details,  # Store the list of dicts
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
        issue_details: List[Dict[str, Any]],  # Changed from issue_numbers
        html_url: str,
    ):
        """Appends a new result (including html_url and issue details) to the issue label cache file."""
        try:
            # Ensure cache directory exists
            self.issue_cache_path.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "repo": repo_full_name,
                "label": label,
                "has_label": has_label,
                "issue_details": issue_details,  # Use new key and store details list
                "html_url": html_url,
                "checked_at": datetime.now(timezone.utc).isoformat(),
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
        """
        Checks if a repository has at least one open issue with the given label.
        NOTE: This method is wrapped by _execute_with_retry in the search loop,
        so direct rate limit handling here is less critical but kept for robustness.
        """
        # Quote the label if it contains spaces
        safe_label = f'"{label}"' if " " in label else label
        issue_query = f"repo:{repo.full_name} is:issue is:open label:{safe_label}"
        try:
            # Search for just one issue to confirm existence
            issues = self.gh.search_issues(query=issue_query)
            return issues.totalCount > 0
        except RateLimitExceededException as rle: # Will be caught by _execute_with_retry
            print(f"Rate limit hit checking issues for {repo.full_name} (will retry).")
            raise rle # Re-raise for _execute_with_retry to handle
        except GithubException as ge:
            # Handle potential 422 errors if the query is complex or repo is weird
            print(f"GitHub error checking issues for {repo.full_name}: {ge}. Skipping.")
            # Non-retryable errors might be raised here, or handled by _execute_with_retry
            raise ge # Re-raise for _execute_with_retry to decide
        except Exception as e:
            print(
                f"Unexpected error checking issues for {repo.full_name}: {e}. Skipping."
            )
            raise e # Re-raise for _execute_with_retry

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
        if exception and hasattr(exception, 'headers') and "Retry-After" in exception.headers:
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
        """Executes a function with exponential backoff for RateLimitExceededException and 5xx GithubException."""
        max_retries = 5
        base_delay = 2  # seconds
        func_name = getattr(func, '__name__', 'unknown_function') # Get function name safely

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitExceededException as rle:
                # Determine limit type based on function name or context if possible
                # Default to 'search' for search_repositories, 'core' otherwise
                limit_type = "search" if "search_repositories" in func_name else "core"
                print(
                    f"\nRate limit hit calling {func_name} (attempt {attempt + 1}/{max_retries}). Waiting..."
                )
                self._wait_for_rate_limit_reset(limit_type, exception=rle)
                # Continue to next attempt after waiting
            except GithubException as ge:
                # Check if it's a server error (5xx) which might be transient
                if hasattr(ge, "status") and ge.status >= 500:
                    delay = (base_delay**attempt) + random.uniform(
                        0, 1
                    )  # Exponential backoff with jitter
                    print(
                        f"\nGitHub API server error calling {func_name} (Status {ge.status}, attempt {attempt + 1}/{max_retries}): {ge.data.get('message', 'No message')}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable GitHub error (e.g., 404, 403, 422)
                    print(f"\nNon-retryable GitHub error calling {func_name}: {ge}")
                    raise  # Re-raise the original exception immediately
            # REMOVED generic `except Exception as e:` block to avoid retrying unexpected errors.
            # Let other exceptions (AttributeError, TypeError, StopIteration, etc.) propagate immediately.

        # This part is reached only if all retries failed for RLE or 5xx GH exceptions
        print(
            f"\nMax retries ({max_retries}) exceeded for function {func_name}. Aborting operation."
        )
        raise RuntimeError(
            f"Failed to execute {func_name} after {max_retries} retries due to persistent API errors."
        )

    def search(
        self,
        *,
        label: str,
        language: str = "python",
        min_stars: int = 20,
        recent_days: int = 365,
        max_results: int = 50,
        cache_file: Path,
        existing_repo_names: Optional[Set[str]] = None,
    ) -> Iterator[Repository]:
        """
        Searches repositories, filters by open issue label, skipping already cached repos,
        yields newly qualified repos, and saves them incrementally to the cache file (JSONL format).
        Handles rate limits with backoff and retry for API calls.
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
        skipped_cached_count = 0

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        checker_context = BrowserIssueChecker() if self.use_browser_checker else None

        try:
            if checker_context:
                checker_context.__enter__()

            # Initial search for repositories - wrap in retry logic
            paginated_list = self._execute_with_retry(
                self.gh.search_repositories,
                query=repo_query,
                sort="updated",
                order="desc",
            )

            # This check is now less critical as _execute_with_retry raises on persistent failure
            # if not paginated_list:
            #     print("[red]Initial repository search failed after multiple retries.")
            #     return # Yield nothing

            print(f"Found {paginated_list.totalCount} potential repos matching base criteria.")
            print(f"Checking them for open '{label}' issues until {max_results} are found...")

            repo_iterator = iter(paginated_list)

            # Open cache file for appending JSON Lines
            with (
                cache_file.open("a", encoding="utf-8") as f_cache,
                tqdm(
                    total=max_results, desc=f"Finding repos w/ '{label}' issues"
                ) as pbar,
            ):
                while found_count < max_results:
                    repo = None # Reset repo for each iteration attempt
                    try:
                        # --- Get next repo --- NOTE: Errors here are handled by outer try-except
                        repo = next(repo_iterator)

                    except StopIteration:
                        print("\nReached end of GitHub repository search results.")
                        break # Exit the while loop cleanly

                    except RateLimitExceededException as rle:
                        # Handle rate limit specifically when getting next item from iterator
                        print(f"\nRate limit hit getting next repo ({rle.status}). Waiting...")
                        # Iteration likely uses 'core' limit (used by underlying get_page)
                        self._wait_for_rate_limit_reset("core", exception=rle)
                        continue # Try getting the *next* item again after waiting

                    except (GithubException, UnknownObjectException) as ge:
                        # Handle other GitHub errors when getting next item
                        print(f"\nGitHub error getting next repo: {ge}. Skipping.")
                        continue # Skip to next repo

                    except Exception as e:
                        # Handle unexpected errors when getting next item
                        print(f"\nUnexpected error getting next repo: {e}. Skipping.")
                        continue # Skip to next repo

                    # --- Process the successfully retrieved repo --- If repo is None, something went wrong above
                    if repo is None:
                        print("[yellow]Warning: Repo object is None after attempting to get next item. Skipping.")
                        continue

                    try:
                        processed_repo_count += 1
                        repo_full_name = repo.full_name
                        pbar.set_postfix_str(f"Checking {repo_full_name}...", refresh=True)

                        # --- Skip if already in the main cache file ---
                        if (
                            existing_repo_names
                            and repo_full_name in existing_repo_names
                        ):
                            pbar.set_postfix_str("Skipping cached", refresh=False)
                            skipped_cached_count += 1
                            continue # Skip to the next repo

                        # --- Check Issue Label Cache (secondary cache) ---
                        has_label = False # Default
                        issue_details = [] # Default
                        cache_key = (repo_full_name, label)
                        cache_hit = cache_key in self.issue_cache

                        if cache_hit:
                            has_label, issue_details = self.issue_cache[cache_key]
                            print(
                                f"  [Cache Hit] Repo: {repo_full_name}, Label: '{label}', Has Label: {has_label}, Issues: {[d.get('number', '?') for d in issue_details]}"
                            )
                        else:
                            # --- Cache Miss - Perform Check --- Wrap check in its own try/except for retry logic failure
                            check_result: Optional[Tuple[bool, List[Dict[str, Any]]]] = None
                            try:
                                if self.use_browser_checker and checker_context:
                                    # Wrap browser check in retry
                                    check_result = self._execute_with_retry(
                                         checker_context.check_repo_for_issue_label,
                                         repo_full_name, label
                                    )
                                else:
                                    # Wrap API check in retry
                                    api_has_label_result = self._execute_with_retry(
                                        self._has_open_issue_with_label, repo, label
                                    )
                                    check_result = (api_has_label_result, []) # API check doesn't return details list

                                # --- Update Cache if check was successful --- (check_result should not be None if no exception from _execute_with_retry)
                                if check_result is not None:
                                    has_label, issue_details_check = check_result
                                    self.issue_cache[cache_key] = (has_label, issue_details_check)
                                    self._append_to_issue_cache(
                                        repo_full_name,
                                        label,
                                        has_label,
                                        issue_details_check,
                                        repo.html_url,
                                    )
                                    # Use the details from the check result
                                    issue_details = issue_details_check
                                    print(
                                        f"  [Check Done] Repo: {repo_full_name}, Label: '{label}', Has Label: {has_label}, Issues: {[d.get('number', '?') for d in issue_details]}. Cached."
                                    )
                                else:
                                     # This case indicates a potential logic error in _execute_with_retry or here
                                     print(f"  [Check Error] Skipping qualification for {repo_full_name} due to unexpected None result from check.")
                                     continue # Skip repo qualification

                            except (RuntimeError, GithubException) as check_err: # Catch errors from _execute_with_retry or non-retryable GH errors during check
                                print(f"  [Check Failed] Skipping qualification for {repo_full_name} due to error: {check_err}")
                                continue # Skip repo qualification
                            except Exception as check_err: # Catch any other unexpected error during check
                                print(f"  [Check Failed] Skipping qualification for {repo_full_name} due to unexpected error: {check_err}")
                                continue # Skip repo qualification

                        # --- Process Qualification --- (has_label is now set either from cache or check)
                        if has_label:
                            found_count += 1
                            pbar.update(1)
                            pbar.set_postfix_str("Found qualified", refresh=False)
                            print(f"\n  [+] Qualified: {repo.full_name}")
                            print(f"      URL: {repo.html_url}")
                            print(f"      ({found_count}/{max_results})")

                            # Add the found issue details to the raw data before saving
                            repo.raw_data["_repobird_found_issues"] = issue_details
                            repo.raw_data.pop("_repobird_found_issue_numbers", None) # Clean up old key if present

                            # Write raw data to cache file immediately
                            json.dump(repo.raw_data, f_cache)
                            f_cache.write("\n")
                            f_cache.flush()

                            yield repo # Yield the qualified repo object

                    except (RateLimitExceededException, GithubException, UnknownObjectException) as inner_e:
                        # Catch errors during repo processing (e.g., cache write, attribute access)
                        print(f"\n[yellow]Warning: Skipping repo {repo_full_name} due to processing error: {inner_e}")
                        if isinstance(inner_e, RateLimitExceededException):
                             # Assume 'core' limit for potential processing API calls (though less likely here)
                             self._wait_for_rate_limit_reset("core", exception=inner_e)
                        continue # Continue to next repo attempt
                    except Exception as inner_e:
                        # Catch unexpected errors during processing
                        repo_name_for_log = repo_full_name if 'repo_full_name' in locals() else (getattr(repo, 'full_name', None) or 'unknown')
                        print(f"\n[red]Unexpected error processing repo {repo_name_for_log}: {inner_e}. Skipping.")
                        continue # Continue to next repo attempt

            # --- End of while loop --- #

            if found_count < max_results:
                print(f"\nWarning: Found only {found_count} repositories meeting all criteria after checking {processed_repo_count} candidates.")
            else:
                print(f"\nSuccessfully found and cached {found_count} repositories meeting all criteria.")

            if skipped_cached_count > 0:
                print(f"Skipped {skipped_cached_count} repositories already present in the cache file.")

        except RuntimeError as e:
            # Catch failure from initial search retry wrapper
            print(f"[red]Error during initial search setup: {e}")
            raise # Re-raise the error <--- MODIFIED HERE
        except GithubException as e:
            # Catch non-retryable errors during initial search setup
            print(f"[red]GitHub API error during initial search setup: {e}")
            if e.status == 401:
                raise RuntimeError("Bad GitHub credentials. Check GITHUB_TOKEN.") from e
            # Re-raise other GithubExceptions if needed, or handle specific statuses
            raise
        except Exception as e:
            # Catch any other unexpected errors during setup
            print(f"[red]An unexpected error occurred during search setup: {e}")
            raise
        finally:
            if checker_context:
                checker_context.__exit__(None, None, None) # Manually exit context

    # These helper methods are less useful now search yields results,
    # but can be kept for potential internal use or adapted if needed.
    # They would need modification to handle the yielding nature of search().
    # def good_first_issue_repos(self, **kwargs) -> Iterator[Repository]:
    #     yield from self.search(label="good first issue", **kwargs)

    # def help_wanted_repos(self, **kwargs) -> Iterator[Repository]:
    #     yield from self.search(label="help wanted", **kwargs)
