from __future__ import annotations
from typing import Iterable, List
from github import Github, Auth  # Added Auth
from github.Repository import Repository
from tqdm import tqdm
from .config import GITHUB_TOKEN, CONCURRENCY
from datetime import datetime, timedelta, timezone # Changed import datetime
import time
import github # Explicitly import github module for exception handling


_QUERY_TEMPLATE = (
    "label:{label} archived:false fork:false stars:>={min_stars}"
    " language:{language} pushed:>{pushed_after}"
)

class GitHubSearcher:
    """Searches GitHub for repositories that fit our outreach criteria."""

    def __init__(self, token: str | None = None):
        # Use Auth.Token for authentication
        auth = Auth.Token(token or GITHUB_TOKEN)
        self.gh = Github(auth=auth, per_page=100, retry=5, timeout=15) # Use Auth.Token, add retry and timeout

    def _build_query(self, *, label: str, language: str, min_stars: int, recent_days: int) -> str:
        # Calculate date for 'pushed_after' using timezone-aware UTC time
        pushed_after_date = datetime.now(timezone.utc) - timedelta(days=recent_days) # Use timezone.utc
        pushed_after_str = pushed_after_date.strftime('%Y-%m-%dT%H:%M:%SZ') # ISO 8601 format
        return _QUERY_TEMPLATE.format(
            label=f'"{label}"', # Quote label if it contains spaces
            language=language,
            min_stars=min_stars,
            pushed_after=pushed_after_str,
        )

    def search(self, *, label: str, language: str = "python", min_stars: int = 20, recent_days: int = 365, max_results: int = 50) -> List[Repository]:
        query = self._build_query(label=label, language=language, min_stars=min_stars, recent_days=recent_days)
        print(f"Searching GitHub with query: {query}") # Log the query
        results: List[Repository] = []
        try:
            paginated_list = self.gh.search_repositories(query=query, sort="updated", order="desc")
            # Respect max_results and handle potential pagination issues
            total_to_fetch = min(max_results, paginated_list.totalCount) if max_results is not None else paginated_list.totalCount
            print(f"Found {paginated_list.totalCount} potential repos, fetching up to {total_to_fetch}...")

            # Ensure total_to_fetch is not None before passing to tqdm
            if total_to_fetch is None:
                print("Warning: Could not determine total count from GitHub. Progress bar might be inaccurate.")
                pbar_total = None
            else:
                pbar_total = total_to_fetch

            with tqdm(total=pbar_total, desc="Searching repos") as pbar:
                 # Iterate manually to control rate limiting and max_results precisely
                 iterator = iter(paginated_list)
                 while total_to_fetch is None or len(results) < total_to_fetch:
                     try:
                         repo = next(iterator)
                         results.append(repo)
                         pbar.update(1)
                         # Break early if max_results is reached (and not None)
                         if max_results is not None and len(results) >= max_results:
                             print(f"Reached max_results ({max_results}). Stopping search.")
                             break
                     except StopIteration:
                         print("Reached end of GitHub search results.")
                         break
                     except github.RateLimitExceededException:
                         print("Rate limit exceeded. Waiting...")
                         limit = self.gh.get_rate_limit().search
                         reset_time = limit.reset.replace(tzinfo=timezone.utc)
                         wait_seconds = (reset_time - datetime.now(timezone.utc)).total_seconds() + 5 # Add buffer
                         print(f"Waiting for {wait_seconds:.0f} seconds until rate limit reset.")
                         time.sleep(max(0, wait_seconds))
                     except github.GithubException as ge:
                         # Handle specific GitHub errors more gracefully
                         print(f"GitHub API error fetching repo: {ge}. Status: {ge.status}. Skipping this repo.")
                         if ge.status == 422: # Unprocessable Entity, often due to complex query
                             print("Query might be too complex or invalid.")
                         # Continue to the next repo if possible
                         continue
                     except Exception as e:
                         print(f"Error fetching repo: {e}. Skipping.")
                         # Optionally break or continue based on error severity
                         # break
                         continue # Continue to next repo

        except github.GithubException as e:
             print(f"GitHub API error during search setup: {e}")
             # Handle specific errors like bad credentials if needed
             if e.status == 401:
                 raise RuntimeError("Bad GitHub credentials. Check GITHUB_TOKEN.") from e
             # Other errors might be transient, return what we have or raise
             # return results
             raise # Re-raise other GitHub errors for now
        except Exception as e:
            print(f"An unexpected error occurred during search: {e}")
            raise # Re-raise unexpected errors

        print(f"Retrieved {len(results)} repositories.")
        return results


    def good_first_issue_repos(self, **kwargs) -> List[Repository]:
        return self.search(label="good first issue", **kwargs) # Use quoted label

    def help_wanted_repos(self, **kwargs) -> List[Repository]:
         return self.search(label="help wanted", **kwargs) # Use quoted label
