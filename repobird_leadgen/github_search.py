from __future__ import annotations
from typing import List
from github import Github, Auth  # Added Auth
from github.Repository import Repository
from tqdm import tqdm
from .config import GITHUB_TOKEN
from datetime import datetime, timedelta, timezone  # Changed import datetime
import time
import github  # Explicitly import github module for exception handling


_QUERY_TEMPLATE = (
    "label:{label} archived:false fork:false stars:>={min_stars}"
    " language:{language} pushed:>{pushed_after}"
)


class GitHubSearcher:
    """Searches GitHub for repositories that fit our outreach criteria."""

    def __init__(self, token: str | None = None):
        # Use Auth.Token for authentication
        auth = Auth.Token(token or GITHUB_TOKEN)
        self.gh = Github(
            auth=auth, per_page=100, retry=5, timeout=15
        )  # Use Auth.Token, add retry and timeout

    def _build_query(
        self, *, label: str, language: str, min_stars: int, recent_days: int
    ) -> str:
        # Calculate date for 'pushed_after' using timezone-aware UTC time
        pushed_after_date = datetime.now(timezone.utc) - timedelta(
            days=recent_days
        )  # Use timezone.utc
        pushed_after_str = pushed_after_date.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )  # ISO 8601 format
        return _QUERY_TEMPLATE.format(
            label=f'"{label}"',  # Quote label if it contains spaces
            language=language,
            min_stars=min_stars,
            pushed_after=pushed_after_str,
        )

    def search(
        self,
        *,
        label: str,
        language: str = "python",
        min_stars: int = 20,
        recent_days: int = 365,
        max_results: int = 50,
    ) -> List[Repository]:
        query = self._build_query(
            label=label, language=language, min_stars=min_stars, recent_days=recent_days
        )
        print(f"Searching GitHub with query: {query}")  # Log the query
        results: List[Repository] = []
        try:
            paginated_list = self.gh.search_repositories(
                query=query, sort="updated", order="desc"
            )
            # Respect max_results and handle potential pagination issues
            total_to_fetch = (
                min(max_results, paginated_list.totalCount)
                if max_results is not None
                else paginated_list.totalCount
            )
            print(
                f"Found {paginated_list.totalCount} potential repos, fetching up to {total_to_fetch}..."
            )

            # Ensure total_to_fetch is not None before passing to tqdm
            if total_to_fetch is None:
                print(
                    "Warning: Could not determine total count from GitHub. Progress bar might be inaccurate."
                )
                pbar_total = None # tqdm handles None total
            else:
                pbar_total = total_to_fetch

            # --- Modified Iteration Logic ---
            # Iterate manually to control rate limiting and max_results precisely within the loop
            repo_iterator = iter(paginated_list)
            processed_count = 0
            # Use tqdm primarily for the progress bar display, but iterate manually
            with tqdm(total=pbar_total, desc="Searching repos") as pbar:
                print(f"!!! DEBUG: pbar object in SUT: {pbar} !!!") # ADDED DEBUG
                while total_to_fetch is None or processed_count < total_to_fetch:
                    try:
                        # Check max_results before fetching next item
                        if max_results is not None and len(results) >= max_results:
                             print(
                                 f"Reached max_results ({max_results}). Stopping search."
                             )
                             break

                        print("!!! DEBUG: Calling next(repo_iterator) !!!") # ADDED DEBUG
                        repo = next(repo_iterator)
                        print(f"!!! DEBUG: Successfully got repo: {repo} !!!") # ADDED DEBUG
                        # We increment processed_count here because next() succeeded
                        processed_count += 1
                        results.append(repo)
                        print("!!! DEBUG: Calling pbar.update(1) after success !!!") # ADDED DEBUG
                        pbar.update(1) # Manually update pbar for the successfully processed item

                    except StopIteration:
                        print("Reached end of GitHub search results.")
                        break # Exit the while loop cleanly
                    except github.RateLimitExceededException:
                        print("!!! DEBUG: Entering RateLimitExceededException block !!!") # ADDED DEBUG PRINT
                        print("Rate limit exceeded during iteration. Waiting...")
                        # Make sure self.gh is accessible here
                        print(f"!!! DEBUG: Calling self.gh.get_rate_limit() on {self.gh} !!!") # ADDED DEBUG
                        limit = self.gh.get_rate_limit().search
                        print(f"!!! DEBUG: Got limit: {limit} !!!") # ADDED DEBUG
                        reset_time = limit.reset.replace(tzinfo=timezone.utc)
                        wait_seconds = (
                            reset_time - datetime.now(timezone.utc)
                        ).total_seconds() + 5  # Add buffer
                        print(
                            f"Waiting for {wait_seconds:.0f} seconds until rate limit reset."
                        )
                        if wait_seconds > 0:
                            time.sleep(wait_seconds)
                        # After waiting, continue the loop to retry fetching the *next* item
                        # The current item that caused the rate limit is skipped implicitly
                        # We don't increment processed_count or update pbar here,
                        # as the item wasn't successfully processed. The loop continues.
                        continue
                    except github.GithubException as ge:
                        print("!!! DEBUG: Entering GithubException block !!!") # ADDED DEBUG PRINT
                        # Handle specific GitHub errors more gracefully
                        print(
                            f"GitHub API error fetching repo details: {ge}. Status: {ge.status}. Skipping this repo."
                        )
                        if ge.status == 422:
                            print("Query might be too complex or invalid.")
                        # Increment processed_count because we attempted to process an item from the iterator
                        processed_count += 1
                        # Update pbar to reflect that an item was processed (even if skipped)
                        print("!!! DEBUG: Calling pbar.update(1) after GithubException !!!") # ADDED DEBUG
                        pbar.update(1)
                        # Continue to the next potential repo
                        continue
                    except Exception as e:
                        print(f"Unexpected error fetching repo details: {e}. Skipping.")
                        # Increment processed_count because we attempted to process an item
                        processed_count += 1
                        # Update pbar
                        print("!!! DEBUG: Calling pbar.update(1) after generic Exception !!!") # ADDED DEBUG
                        pbar.update(1)
                        # Continue to next repo
                        continue
            # --- End Modified Iteration Logic ---

        except github.GithubException as e:
            # This catches errors during the initial search_repositories call
            print(f"GitHub API error during initial search setup: {e}")
            if e.status == 401:
                raise RuntimeError("Bad GitHub credentials. Check GITHUB_TOKEN.") from e
            # Re-raise other GitHub errors during setup
            raise
        except Exception as e:
            # Catch any other unexpected errors during setup
            print(f"An unexpected error occurred during search setup: {e}")
            raise

        print(f"Retrieved {len(results)} repositories.")
        return results

    def good_first_issue_repos(self, **kwargs) -> List[Repository]:
        return self.search(label="good first issue", **kwargs)  # Use quoted label

    def help_wanted_repos(self, **kwargs) -> List[Repository]:
        return self.search(label="help wanted", **kwargs)  # Use quoted label
