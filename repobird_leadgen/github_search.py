from __future__ import annotations

import json  # For saving raw data incrementally

# from rich.console import Console # Remove Console import
import logging  # Import logging
import random  # For jitter
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path  # For cache file path handling
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)  # Added Set, List, Any

from github import (
    Auth,
    Github,
    GithubException,
    RateLimitExceededException,
    UnknownObjectException,
)
from github.Issue import Issue
from github.Repository import Repository
from tqdm import tqdm

# Import the new browser checker
from .browser_checker import BrowserIssueChecker
from .config import CACHE_DIR, GITHUB_TOKEN  # Added CACHE_DIR

# Base query for repositories, label is handled separately
_REPO_QUERY_TEMPLATE = (
    "archived:false fork:false stars:>={min_stars}"
    " language:{language} pushed:>{pushed_after}"
)

# Get logger for this module
logger = logging.getLogger(__name__)


class GitHubSearcher:
    """
    Searches GitHub for repositories using the API, filters by issue criteria
    (label, age, linked PRs), and optionally checks labels using a browser.
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
        # Increased timeout for potentially longer issue/timeline calls
        self.gh = Github(auth=auth, per_page=100, retry=5, timeout=30)
        self.use_browser_checker = use_browser_checker
        self._browser_checker_instance: Optional[BrowserIssueChecker] = None

        # Issue Label Cache (stores basic label existence check result)
        self.issue_cache_path = Path(CACHE_DIR) / "issue_label_cache.jsonl"
        # Cache stores: (repo_full_name, label) -> (has_label_bool, list_of_basic_issue_details)
        self.issue_cache: Dict[
            Tuple[str, str],
            Tuple[bool, List[Dict[str, Any]]],
        ] = {}
        self._load_issue_cache()

        logger.info(
            f"GitHubSearcher Initialized. Using Browser Checker: {self.use_browser_checker}. Issue cache loaded with {len(self.issue_cache)} entries."
        )

    def _load_issue_cache(self):
        """Loads the basic issue label check results from the cache file."""
        if not self.issue_cache_path.exists():
            logger.info(
                f"Issue cache file not found: {self.issue_cache_path}. Starting fresh."
            )
            return

        logger.info(f"Loading issue cache from: {self.issue_cache_path}")
        count = 0
        try:
            with self.issue_cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "repo" in data and "label" in data and "has_label" in data:
                            # Ensure issue_details is always a list, default to empty if missing
                            issue_details = data.get("issue_details", [])
                            if not isinstance(issue_details, list):
                                logger.warning(
                                    f"Invalid 'issue_details' format in cache for {data['repo']}, resetting to []."
                                )
                                issue_details = []

                            self.issue_cache[(data["repo"], data["label"])] = (
                                data["has_label"],
                                issue_details,
                            )
                            count += 1
                        else:
                            logger.warning(
                                f"Skipping invalid line in issue cache: {line[:100]}..."
                            )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping invalid JSON line in issue cache: {line[:100]}..."
                        )
            logger.info(f"Loaded {count} entries into issue cache.")
        except Exception as e:
            logger.error(f"Error loading issue cache file {self.issue_cache_path}: {e}")
            self.issue_cache = {}  # Start fresh if loading fails

    def _append_to_issue_cache(
        self,
        repo_full_name: str,
        label: str,
        has_label: bool,
        issue_details: List[Dict[str, Any]],
        html_url: str,
    ):
        """Appends a basic label check result to the issue label cache file."""
        try:
            self.issue_cache_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "repo": repo_full_name,
                "label": label,
                "has_label": has_label,
                "issue_details": issue_details,
                "html_url": html_url,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            with self.issue_cache_path.open("a", encoding="utf-8") as f:
                json.dump(entry, f)
                f.write("\n")
        except Exception as e:
            logger.error(
                f"Error appending to issue cache file {self.issue_cache_path}: {e}"
            )

    def _build_repo_query(
        self, *, language: str, min_stars: int, recent_days: int
    ) -> str:
        """Builds the query for the initial repository search."""
        pushed_after_date = datetime.now(timezone.utc) - timedelta(days=recent_days)
        pushed_after_str = pushed_after_date.strftime("%Y-%m-%d")
        return _REPO_QUERY_TEMPLATE.format(
            language=language,
            min_stars=min_stars,
            pushed_after=pushed_after_str,
        )

    def _has_open_issue_with_label_api(
        self, repo: Repository, label: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        API check: Does the repo have open issues with the label?
        Returns (bool, list_of_issue_details), where details are {'number': int, 'html_url': str}.
        Fetches details for up to 30 issues if found.
        """
        safe_label = f'"{label}"' if " " in label else label
        issue_query = f"repo:{repo.full_name} is:issue is:open label:{safe_label}"
        issue_details: List[Dict[str, Any]] = []
        try:
            # Use execute_with_retry for the search call itself
            issues_paginator = self._execute_with_retry(
                self.gh.search_issues, query=issue_query
            )
            if not issues_paginator:
                logger.warning(
                    f"API check for label '{label}' on {repo.full_name} failed to get paginator after retries. Assuming false."
                )
                return False, []

            has_issues = issues_paginator.totalCount > 0
            if has_issues:
                # Fetch details for the first few issues (up to 30)
                count = 0
                max_to_fetch = 30
                for issue in issues_paginator:
                    if count >= max_to_fetch:
                        break
                    issue_details.append(
                        {"number": issue.number, "html_url": issue.html_url}
                    )
                    count += 1
            return has_issues, issue_details
        except RuntimeError as e:
            # Catch failure after retries from _execute_with_retry
            logger.warning(
                f"API check for label '{label}' on {repo.full_name} failed after retries: {e}. Assuming false."
            )
            return False, []
        except GithubException as ge:
            logger.warning(
                f"GitHub error during API label check for {repo.full_name}: {ge}. Assuming false."
            )
            return False, []
        except Exception as e:
            logger.warning(
                f"Unexpected error during API label check for {repo.full_name}: {e}. Assuming false."
            )
            return False, []

    # --- New Helper Methods --- START
    def _get_linked_prs_count(self, issue: Issue) -> int:
        """Counts linked PRs for an issue using the timeline API."""
        count = 0
        try:
            # Wrap the timeline fetching in retry logic
            timeline = self._execute_with_retry(issue.get_timeline)
            if not timeline:
                logger.warning(
                    f"Failed to get timeline for issue #{issue.number} after retries."
                )
                return -1  # Indicate error

            for event in timeline:
                # Check event type and cross-referenced source
                if (
                    event.event == "cross-referenced"
                    and event.source
                    and hasattr(event.source, "issue")
                    and hasattr(event.source.issue, "pull_request")
                    and event.source.issue.pull_request is not None
                ):
                    count += 1
            return count
        except RuntimeError as e:
            logger.warning(
                f"Timeline fetch for issue {issue.number} in repo {issue.repository.full_name} failed after retries: {e}. Cannot count PRs."
            )
            return -1  # Indicate error
        except GithubException as ge:
            logger.warning(
                f"GitHub error fetching timeline for issue {issue.number} in repo {issue.repository.full_name}: {ge}. Cannot count PRs."
            )
            return -1
        except Exception as e:
            logger.warning(
                f"Unexpected error fetching timeline for issue {issue.number} in repo {issue.repository.full_name}: {e}. Cannot count PRs."
            )
            return -1

    def _find_qualifying_issues(
        self,
        repo: Repository,
        label: str,
        max_issue_age_days: Optional[int],
        # max_linked_prs: Optional[int], # Removed PR filter from this function's responsibility
    ) -> List[Dict[str, Any]]:
        """
        Fetches open issues with the label and returns details (number, url) of those
        that satisfy the age criteria (up to a limit).
        PR filtering is handled separately in the main search loop.
        Returns a list of dictionaries, each containing 'number' and 'html_url'.
        """
        candidate_issues_details: List[Dict[str, Any]] = []  # Renamed variable
        try:
            logger.info(
                f"Performing detailed check (Age Filter Only): Repo {repo.full_name} ({repo.html_url}), Label '{label}', Max Age: {max_issue_age_days}"
            )
            # Fetch issues with the label using the API (wrapped in retry)
            issues_paginator = self._execute_with_retry(
                repo.get_issues, state="open", labels=[label]
            )

            if not issues_paginator:
                logger.warning(
                    f"Failed to fetch issues for {repo.full_name} after retries."
                )
                return []  # Return empty list on failure

            # Calculate age threshold if needed
            age_threshold = None
            if max_issue_age_days is not None:
                age_threshold = datetime.now(timezone.utc) - timedelta(
                    days=max_issue_age_days
                )

            # Iterate through issues, checking criteria
            # Limit iteration to avoid excessive API calls if many issues exist
            issues_checked = 0
            max_issues_to_check = 300  # Configurable? Maybe later.

            # --- Add tqdm progress bar for issue checking ---
            total_issues_with_label = issues_paginator.totalCount
            progress_total = min(total_issues_with_label, max_issues_to_check)
            issue_iterator = iter(issues_paginator)  # Get iterator

            logger.info(
                f"Found {total_issues_with_label} issues with label for {repo.full_name}, checking up to {max_issues_to_check}"
            )

            with tqdm(
                total=progress_total, desc="  Checking issues", leave=False
            ) as issue_pbar:
                for issue in issue_iterator:
                    issues_checked += 1
                    if issues_checked > max_issues_to_check:
                        logger.info(
                            f"Checked {max_issues_to_check} issues for {repo.full_name}, none qualified so far. Stopping detailed check."
                        )
                        break  # Stop checking after a limit

                    # Update progress bar description with current issue URL
                    issue_pbar.set_postfix_str(f"Issue {issue.html_url}", refresh=True)

                    logger.debug(f"Checking issue: {issue.html_url}")  # Use debug level

                    # 1. Check Age
                    if age_threshold:
                        # Ensure issue.created_at is timezone-aware (it should be from PyGithub)
                        created_at_aware = issue.created_at
                        if created_at_aware.tzinfo is None:
                            # Should not happen with PyGithub, but handle defensively
                            created_at_aware = created_at_aware.replace(
                                tzinfo=timezone.utc
                            )

                        if created_at_aware < age_threshold:
                            logger.info(
                                f"Issue {issue.html_url} is too old (created {created_at_aware.date()}). Skipping."
                            )
                            issue_pbar.update(1)  # Advance progress bar
                            continue  # Skip to next issue

                    # 2. PR Check Removed: This is now handled after this function returns.

                    # If we reach here, the issue satisfies the age criteria (if applied)
                    logger.info(f"Issue {issue.html_url} meets age criteria.")
                    candidate_issues_details.append(
                        {"number": issue.number, "html_url": issue.html_url}
                    )
                    issue_pbar.update(1)  # Advance progress bar
                    # Continue checking other issues up to the limit

            # If loop finishes, return the list of candidate details found
            logger.info(
                f"Found {len(candidate_issues_details)} candidate issues (meeting age criteria) among the first {issues_checked} checked for {repo.full_name} with label '{label}'."
            )
            return candidate_issues_details  # Return candidates

        except RuntimeError as e:
            logger.error(
                f"Error during detailed issue check for {repo.full_name} after retries: {e}"
            )
            return []  # Return empty list on error
        except GithubException as ge:
            logger.error(
                f"GitHub error during detailed issue check for {repo.full_name}: {ge}"
            )
            return []  # Return empty list on error
        except Exception as e:
            logger.exception(
                f"Unexpected error during detailed issue check for {repo.full_name}: {e}"
            )
            return []  # Return empty list on error

    # --- New Helper Methods --- END

    def _wait_for_rate_limit_reset(
        self,
        limit_type: str = "search",
        exception: Optional[RateLimitExceededException] = None,
    ):
        """Waits until the GitHub API rate limit is reset."""
        wait_seconds = 60  # Default
        reset_time_str = "unknown"
        remaining_str = "unknown"
        limit_str = "unknown"

        # 1. Retry-After Header
        if exception and "Retry-After" in exception.headers:
            try:
                wait_seconds = int(exception.headers["Retry-After"]) + 5
                logger.warning(
                    f"Using Retry-After header: waiting {wait_seconds:.0f} seconds."
                )
                time.sleep(wait_seconds)
                return
            except (ValueError, TypeError):
                logger.warning("Could not parse Retry-After header.")

        # 2. Primary Rate Limit Info
        try:
            limits = self.gh.get_rate_limit()
            limit_map = {"search": limits.search, "core": limits.core}
            # Add other limits if needed (graphql, integration_manifest, code_scanning_upload)
            limit_data = limit_map.get(limit_type)

            if limit_data:
                reset_time_utc = limit_data.reset.replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                calculated_wait = (reset_time_utc - now_utc).total_seconds() + 5
                reset_time_str = reset_time_utc.isoformat()
                remaining_str = str(limit_data.remaining)
                limit_str = str(limit_data.limit)

                if calculated_wait > 0:
                    wait_seconds = calculated_wait
                else:
                    # Limit should be reset, but add safety delay
                    wait_seconds = 5
                    logger.info(
                        f"Primary '{limit_type}' limit should be reset. Waiting safety {wait_seconds}s."
                    )

            else:
                logger.warning(
                    f"Unknown rate limit type '{limit_type}'. Using default wait."
                )
                wait_seconds = 60  # Fallback

        except Exception as e:
            logger.error(
                f"Error getting rate limit details: {e}. Falling back to 60s wait."
            )
            wait_seconds = 60  # Fallback

        logger.warning(
            f"Waiting {wait_seconds:.0f} seconds for '{limit_type}' limit (Reset: {reset_time_str}, Remaining: {remaining_str}/{limit_str})."
        )
        time.sleep(wait_seconds)

    def _execute_with_retry(self, func, *args, **kwargs):
        """Executes a function with exponential backoff for specific GitHub errors."""
        max_retries = 5
        base_delay = 3  # Increased base delay slightly
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitExceededException as rle:
                # Determine limit type based on function name or common patterns
                func_name = func.__name__
                limit_type = "core"  # Default to core, most API calls use this
                if "search" in func_name:
                    limit_type = "search"
                elif "get_timeline" in func_name:
                    limit_type = "core"
                elif "get_issues" in func_name:
                    limit_type = "core"

                logger.warning(
                    f"Rate limit hit ({limit_type}, attempt {attempt + 1}/{max_retries}). Waiting..."
                )
                self._wait_for_rate_limit_reset(limit_type, exception=rle)
            except GithubException as ge:
                if hasattr(ge, "status") and ge.status >= 500:
                    delay = (base_delay**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"GitHub API server error (Status {ge.status}, attempt {attempt + 1}/{max_retries}): {ge.data.get('message', 'No message')}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                elif (
                    hasattr(ge, "status")
                    and ge.status == 403
                    and "secondary rate limit" in str(ge.data).lower()
                ):
                    # Handle secondary rate limits (often lack Retry-After)
                    delay = (base_delay ** (attempt + 1)) + random.uniform(
                        0, 5
                    )  # Longer, more random delay for secondary
                    logger.warning(
                        f"GitHub API secondary rate limit detected (attempt {attempt + 1}/{max_retries}). Retrying aggressively in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                # Handle 404 Not Found gracefully for specific operations if needed
                elif (
                    hasattr(ge, "status")
                    and ge.status == 404
                    and func.__name__ in ["get_issues", "get_timeline"]
                ):
                    logger.warning(
                        f"Resource not found (404) for {func.__name__}. Returning None."
                    )
                    return None  # Allow flow to continue if a resource isn't found
                else:
                    logger.error(f"Non-retryable GitHub error encountered: {ge}")
                    raise  # Re-raise other GithubExceptions
            except Exception as e:
                # Catch other exceptions like connection errors etc.
                logger.error(
                    f"Unexpected error during API call (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    delay = (base_delay**attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error("Max retries reached for unexpected error.")
                    raise  # Re-raise after max retries

        logger.error(
            f"Max retries ({max_retries}) exceeded for function {func.__name__}. Aborting operation."
        )
        raise RuntimeError(
            f"Failed to execute {func.__name__} after {max_retries} retries due to persistent API errors."
        )

    def search(
        self,
        *,
        label: str,
        language: str = "python",
        min_stars: int = 20,
        recent_days: int = 365,
        max_results: int = 50,
        max_issue_age_days: Optional[int] = None,  # New filter
        max_linked_prs: Optional[int] = None,  # New filter
        cache_file: Path,
        existing_repo_names: Optional[Set[str]] = None,
    ) -> Iterator[Repository]:
        """
        Searches repositories, filters by label, issue age, linked PRs, skipping cached repos,
        yields newly qualified repos, and saves them incrementally to the cache file.
        Handles rate limits with backoff and retry.

        Args:
            label: Issue label to filter for.
            language: Programming language filter.
            min_stars: Minimum stars for repository.
            recent_days: Minimum recency of pushes (days).
            max_results: Maximum number of qualified repositories to yield.
            max_issue_age_days: Optional. Max age of qualifying issues in days.
            max_linked_prs: Optional. Max number of linked PRs for qualifying issues.
            cache_file: Path to the output cache file storing qualified repo data (JSONL).
            existing_repo_names: Optional set of repo names already in the cache file to skip.

        Yields:
            Qualified Repository objects.
        """
        repo_query = self._build_repo_query(
            language=language, min_stars=min_stars, recent_days=recent_days
        )
        logger.info(f"Searching GitHub repositories with query: {repo_query}")
        logger.info(f"Filtering for repos with open issues labeled: '{label}'")
        if max_issue_age_days is not None:
            logger.info(f"Applying Max Issue Age filter: {max_issue_age_days} days")
        if max_linked_prs is not None:
            logger.info(f"Applying Max Linked PRs filter: {max_linked_prs}")
        logger.info(f"Saving results incrementally to: {cache_file}")
        if self.use_browser_checker:
            logger.warning(
                "Using Browser Checker for initial label check (ignored if detailed filters active)."
            )

        found_count = 0
        processed_repo_count = 0
        skipped_cached_count = 0
        detailed_check_count = 0  # Count how many repos needed the deeper check

        # Determine if detailed filtering is active
        detailed_filters_active = (
            max_issue_age_days is not None or max_linked_prs is not None
        )

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Initialize browser checker if the flag is set
        checker_context = None
        browser_checker_instance = (
            None  # Keep track of the instance for context management
        )
        if self.use_browser_checker:
            logger.info("Browser checker requested, initializing...")
            browser_checker_instance = BrowserIssueChecker()
            checker_context = (
                browser_checker_instance.__enter__()
            )  # Enter context immediately

        try:
            # No need to enter context again here

            paginated_list = self._execute_with_retry(
                self.gh.search_repositories,
                query=repo_query,
                sort="updated",
                order="desc",
            )

            if not paginated_list:
                logger.error("Initial repository search failed after multiple retries.")
                return

            logger.info(
                f"Found {paginated_list.totalCount} potential repos matching base criteria."
            )
            logger.info(
                f"Checking them for qualifying '{label}' issues until {max_results} are found..."
            )

            repo_iterator = iter(paginated_list)

            with (
                cache_file.open("a", encoding="utf-8") as f_cache,
                tqdm(
                    total=max_results, desc=f"Finding repos w/ '{label}' issues"
                ) as pbar,
            ):
                while found_count < max_results:
                    try:
                        repo = self._execute_with_retry(next, repo_iterator)
                        processed_repo_count += 1
                        repo_full_name = repo.full_name
                        pbar.set_postfix_str(
                            f"Checking {repo_full_name}...", refresh=True
                        )

                        # --- Skip if already in the main output cache file ---
                        if (
                            existing_repo_names
                            and repo_full_name in existing_repo_names
                        ):
                            pbar.set_postfix_str("Skipping cached", refresh=False)
                            skipped_cached_count += 1
                            continue

                        # --- Qualification Logic --- START
                        repo_qualifies = False
                        found_issues_list: List[
                            Dict[str, Any]
                        ] = []  # Final list of qualifying issues for this repo

                        if detailed_filters_active:
                            # --- Detailed Filter Path (API Age Check -> Optional PR Check) ---
                            pbar.set_postfix_str(
                                f"API Age check {repo_full_name}", refresh=True
                            )
                            detailed_check_count += 1
                            # 1. Find candidates based on AGE filter only
                            candidate_issues = self._find_qualifying_issues(
                                repo,
                                label,
                                max_issue_age_days,  # Pass only age filter
                            )

                            if not candidate_issues:
                                logger.info(
                                    f"Repo {repo_full_name} skipped (no issues met age criteria)."
                                )
                                repo_qualifies = False
                            elif max_linked_prs == 0:
                                # 2a. Handle max_linked_prs == 0 (API or Browser)
                                candidate_numbers = [
                                    item["number"]
                                    for item in candidate_issues
                                    if item.get("number")
                                ]
                                if not candidate_numbers:
                                    logger.warning(
                                        f"Repo {repo_full_name}: Age check found candidates, but failed to extract numbers."
                                    )
                                    repo_qualifies = False  # Treat as non-qualifying if numbers missing
                                elif self.use_browser_checker and checker_context:
                                    # --- Hybrid Browser PR Check ---
                                    pbar.set_postfix_str(
                                        f"Browser PR check {repo_full_name}",
                                        refresh=True,
                                    )
                                    # Pass the label and age to the browser verification function
                                    verified_numbers = (
                                        checker_context.verify_issues_no_prs(
                                            repo_full_name,
                                            label,  # Pass label
                                            max_issue_age_days,  # Pass age filter
                                        )
                                    )
                                    # Use the verified numbers directly to construct the final list
                                    found_issues_list = [
                                        {
                                            "number": num,
                                            "html_url": f"{repo.html_url}/issues/{num}",
                                        }
                                        for num in verified_numbers
                                    ]
                                    repo_qualifies = bool(found_issues_list)
                                    if repo_qualifies:
                                        # Log the qualifying repo and the specific issue URLs found
                                        issue_urls_str = ", ".join(
                                            [
                                                f"#{item['number']} ({item['html_url']})"
                                                for item in found_issues_list
                                            ]
                                        )
                                        logger.info(
                                            f"Repo {repo_full_name} qualifies (Browser PR Check): Found {len(found_issues_list)} issues: {issue_urls_str}"
                                        )
                                    else:
                                        logger.info(
                                            f"Repo {repo_full_name} skipped: Browser verification found no issues with 0 linked PRs among candidates."
                                        )
                                else:
                                    # --- API PR Check (max_linked_prs == 0) ---
                                    pbar.set_postfix_str(
                                        f"API PR check {repo_full_name}", refresh=True
                                    )
                                    final_issues_for_repo = []
                                    issues_processed_for_prs = 0
                                    with tqdm(
                                        total=len(candidate_issues),
                                        desc="  Checking PRs (API)",
                                        leave=False,
                                    ) as pr_pbar:
                                        for issue_detail in candidate_issues:
                                            issue_num = issue_detail.get("number")
                                            if not issue_num:
                                                continue  # Should not happen if _find_qualifying_issues worked

                                            # Need to get the Issue object to check timeline
                                            try:
                                                issue_obj = self._execute_with_retry(
                                                    repo.get_issue, number=issue_num
                                                )
                                                if not issue_obj:
                                                    raise UnknownObjectException(
                                                        404, "Not Found", {}
                                                    )  # Treat retry failure as not found

                                                pr_count = self._get_linked_prs_count(
                                                    issue_obj
                                                )
                                                issues_processed_for_prs += 1
                                                pr_pbar.update(1)
                                                if pr_count == 0:
                                                    logger.info(
                                                        f"Issue #{issue_num} kept (0 linked PRs)."
                                                    )
                                                    final_issues_for_repo.append(
                                                        issue_detail
                                                    )
                                                elif pr_count > 0:
                                                    logger.info(
                                                        f"Issue #{issue_num} skipped ({pr_count} linked PRs)."
                                                    )
                                                else:  # pr_count == -1 (error)
                                                    logger.warning(
                                                        f"Issue #{issue_num} skipped (error checking PRs)."
                                                    )

                                            except (
                                                GithubException,
                                                RuntimeError,
                                                UnknownObjectException,
                                            ) as e:
                                                logger.warning(
                                                    f"Skipping PR check for issue #{issue_num} due to API error: {e}"
                                                )
                                                pr_pbar.update(
                                                    1
                                                )  # Still advance progress
                                                continue  # Skip this issue if we can't check its PRs

                                    found_issues_list = final_issues_for_repo
                                    repo_qualifies = bool(found_issues_list)
                                    if repo_qualifies:
                                        logger.info(
                                            f"Repo {repo_full_name} qualifies: Found {len(found_issues_list)} issues meeting age and 0-PR criteria via API."
                                        )
                                    else:
                                        logger.info(
                                            f"Repo {repo_full_name} skipped: API check found no issues with 0 linked PRs among candidates."
                                        )

                            elif (
                                max_linked_prs is not None
                            ):  # Handle max_linked_prs > 0
                                # 2b. API PR Check (max_linked_prs > 0)
                                pbar.set_postfix_str(
                                    f"API PR check {repo_full_name}", refresh=True
                                )
                                final_issues_for_repo = []
                                with tqdm(
                                    total=len(candidate_issues),
                                    desc="  Checking PRs (API)",
                                    leave=False,
                                ) as pr_pbar:
                                    for issue_detail in candidate_issues:
                                        issue_num = issue_detail.get("number")
                                        if not issue_num:
                                            continue

                                        try:
                                            issue_obj = self._execute_with_retry(
                                                repo.get_issue, number=issue_num
                                            )
                                            if not issue_obj:
                                                raise UnknownObjectException(
                                                    404, "Not Found", {}
                                                )

                                            pr_count = self._get_linked_prs_count(
                                                issue_obj
                                            )
                                            pr_pbar.update(1)
                                            if (
                                                pr_count != -1
                                                and pr_count <= max_linked_prs
                                            ):
                                                logger.info(
                                                    f"Issue #{issue_num} kept ({pr_count} <= {max_linked_prs} linked PRs)."
                                                )
                                                final_issues_for_repo.append(
                                                    issue_detail
                                                )
                                            elif pr_count > max_linked_prs:
                                                logger.info(
                                                    f"Issue #{issue_num} skipped ({pr_count} > {max_linked_prs} linked PRs)."
                                                )
                                            else:  # pr_count == -1 (error)
                                                logger.warning(
                                                    f"Issue #{issue_num} skipped (error checking PRs)."
                                                )

                                        except (
                                            GithubException,
                                            RuntimeError,
                                            UnknownObjectException,
                                        ) as e:
                                            logger.warning(
                                                f"Skipping PR check for issue #{issue_num} due to API error: {e}"
                                            )
                                            pr_pbar.update(1)
                                            continue

                                found_issues_list = final_issues_for_repo
                                repo_qualifies = bool(found_issues_list)
                                if repo_qualifies:
                                    logger.info(
                                        f"Repo {repo_full_name} qualifies: Found {len(found_issues_list)} issues meeting age and PR criteria (<= {max_linked_prs}) via API."
                                    )
                                else:
                                    logger.info(
                                        f"Repo {repo_full_name} skipped: API check found no issues meeting PR criteria (<= {max_linked_prs}) among candidates."
                                    )

                            else:
                                # 2c. No PR filter applied, candidates from age check are final
                                found_issues_list = candidate_issues
                                repo_qualifies = True  # Qualifies if candidates exist
                                logger.info(
                                    f"Repo {repo_full_name} qualifies: Found {len(found_issues_list)} issues meeting age criteria (no PR filter)."
                                )

                        else:
                            # --- Basic Filter Path (No detailed filters) ---
                            # This path uses either Browser Checker OR basic API check + cache
                            if self.use_browser_checker and checker_context:
                                # --- Basic Browser Check ---
                                pbar.set_postfix_str(
                                    f"Browser check {repo_full_name}", refresh=True
                                )
                                try:
                                    # Browser checker handles label check internally, pass age filter
                                    repo_qualifies, browser_issue_details = (
                                        checker_context.check_repo_for_issue_label(
                                            repo_full_name,
                                            label,
                                            None,  # No PR filter needed here, handled by query if max_linked_prs=0
                                            max_issue_age_days,  # Pass age filter
                                        )
                                    )
                                    found_issues_list = [
                                        {
                                            "number": detail.get("number"),
                                            "html_url": f"{repo.html_url}/issues/{detail.get('number')}"
                                            if detail.get("number")
                                            else None,
                                        }
                                        for detail in browser_issue_details
                                        if detail.get("number") is not None
                                    ]
                                except Exception as browser_err:
                                    logger.error(
                                        f"Error during browser check for {repo_full_name}: {browser_err}. Skipping repo."
                                    )
                                    continue
                            else:
                                # --- Basic API Check + Cache ---
                                cache_key = (repo_full_name, label)
                            issue_details_for_cache: List[Dict[str, Any]] = []

                            if cache_key in self.issue_cache:
                                repo_qualifies, issue_details_for_cache = (
                                    self.issue_cache[cache_key]
                                )
                                logger.info(
                                    f"Cache Hit: Repo: {repo_full_name}, Label: '{label}', Has Label: {repo_qualifies}"
                                )
                            else:
                                logger.info(
                                    f"Cache Miss: API Checking {repo_full_name} for '{label}'..."
                                )
                                try:
                                    repo_qualifies, issue_details_for_cache = (
                                        self._has_open_issue_with_label_api(repo, label)
                                    )
                                    self.issue_cache[cache_key] = (
                                        repo_qualifies,
                                        issue_details_for_cache,
                                    )
                                    self._append_to_issue_cache(
                                        repo_full_name,
                                        label,
                                        repo_qualifies,
                                        issue_details_for_cache,
                                        repo.html_url,
                                    )
                                    logger.info(
                                        f"API Check Done: Repo: {repo_full_name}, Label: '{label}', Has Label: {repo_qualifies}. Cached."
                                    )
                                except Exception as api_err:
                                    logger.error(
                                        f"Error during API label check for {repo_full_name}: {api_err}. Skipping repo."
                                    )
                                    continue  # Skip to next repo

                            # Extract number and URL for found_issues_list from the basic check details
                            found_issues_list = [
                                {
                                    "number": item.get("number"),
                                    "html_url": item.get("html_url"),
                                }
                                for item in issue_details_for_cache
                                if isinstance(item, dict)
                                and item.get("number") is not None
                            ]

                        # --- Qualification Logic --- END

                        # --- Process Qualified Repo ---
                        # Use the repo_qualifies flag determined by the initial check (cache or live)
                        if repo_qualifies:
                            found_count += 1
                            pbar.update(1)
                            pbar.set_postfix_str("Found qualified", refresh=False)
                            # Log qualification info
                            logger.info(
                                f"Qualified: {repo.full_name} ({repo.html_url})"
                            )
                            logger.info("  Found Issues:")
                            for issue_detail in found_issues_list:
                                issue_num = issue_detail.get("number")
                                issue_url = issue_detail.get("html_url")
                                # Construct URL if missing (e.g., from basic API check path)
                                if not issue_url and issue_num:
                                    issue_url = f"{repo.html_url}/issues/{issue_num}"
                                logger.info(f"    - Issue #{issue_num}: {issue_url}")
                            logger.info(f"  (Count: {found_count}/{max_results})")

                            # Save raw data to the main cache file
                            # Add the list of found issue details (number and URL)
                            # Ensure URL is present in the saved data
                            issues_to_save = []
                            for issue_detail in found_issues_list:
                                issue_num = issue_detail.get("number")
                                issue_url = issue_detail.get("html_url")
                                if not issue_url and issue_num:
                                    issue_url = f"{repo.html_url}/issues/{issue_num}"
                                if issue_num and issue_url:  # Only save if we have both
                                    issues_to_save.append(
                                        {"number": issue_num, "html_url": issue_url}
                                    )

                            repo.raw_data["found_issues"] = (
                                issues_to_save  # Store list of dicts
                            )

                            # Remove the old key if it exists (defensive)
                            repo.raw_data.pop("_repobird_found_issues_basic", None)

                            json.dump(repo.raw_data, f_cache)
                            f_cache.write("\n")
                            f_cache.flush()
                            yield repo
                        else:
                            # Failed qualification (either basic or detailed)
                            reason = (
                                "failed detailed filter"
                                if detailed_filters_active
                                else "no matching label"
                            )
                            pbar.set_postfix_str(f"Skipped ({reason})", refresh=False)
                            if detailed_filters_active:
                                logger.info(
                                    f"Repo {repo_full_name} skipped (failed detailed age/PR filter)."
                                )
                            # No need for else log, basic filter path logs its own messages

                    except StopIteration:
                        logger.info("Reached end of GitHub repository search results.")
                        break
                    except (RuntimeError, GithubException, UnknownObjectException) as e:
                        logger.warning(
                            f"Skipping repo {repo_full_name if 'repo' in locals() else '(unknown)'} due to error: {e}"
                        )
                        if (
                            isinstance(e, GithubException)
                            and hasattr(e, "status")
                            and e.status == 422
                        ):
                            logger.error(
                                "Stopping search due to persistent invalid query (422)."
                            )
                            break
                        continue  # Skip to next repo
                    except Exception as e:
                        # import traceback # No longer needed with logger.exception
                        logger.exception(
                            f"Unexpected error processing repo stream: {e}. Skipping."
                        )
                        # traceback.print_exc() # logger.exception includes traceback
                        continue  # Skip to next repo

            # End of loop summary
            pbar.close()  # Ensure progress bar is closed cleanly
            logger.info(
                f"Search loop finished. Processed: {processed_repo_count}, Found Qualified: {found_count}"
            )
            if skipped_cached_count > 0:
                logger.info(
                    f"Skipped {skipped_cached_count} repositories already in main cache."
                )
            if detailed_filters_active:
                logger.info(
                    f"Performed detailed age/PR check on {detailed_check_count} repositories."
                )

            if (
                found_count < max_results
                and processed_repo_count < paginated_list.totalCount
            ):
                logger.info(
                    f"Stopped after finding {found_count} qualified repositories (max_results reached)."
                )
            elif found_count < max_results:
                logger.warning(
                    f"Found only {found_count} qualified repositories after checking all {processed_repo_count} processed repos."
                )
            elif found_count == 0:
                logger.warning(
                    f"No repositories matching all criteria (including detailed filters if active) were found among {processed_repo_count} processed repos."
                )
            else:
                logger.info(
                    f"Successfully found and cached {found_count} qualified repositories."
                )

        except RuntimeError as e:
            logger.error(f"Error during search operation: {e}")
        except GithubException as e:
            logger.error(f"GitHub API error during search setup: {e}")
            if hasattr(e, "status") and e.status == 401:
                raise RuntimeError("Bad GitHub credentials. Check GITHUB_TOKEN.") from e
        except Exception as e:
            # import traceback # No longer needed
            logger.exception(f"An unexpected error occurred during search setup: {e}")
            # traceback.print_exc() # logger.exception includes traceback
            raise
        finally:
            # Exit browser context if it was initialized
            if browser_checker_instance:
                logger.info("Shutting down browser checker...")
                browser_checker_instance.__exit__(None, None, None)
