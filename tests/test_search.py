import pytest
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json # For checking JSON content

from github import (
    Github,
    Auth,
    RateLimitExceededException,
    UnknownObjectException,
    GithubException,
)
from github.Repository import Repository
from github.ContentFile import ContentFile
from github.PaginatedList import PaginatedList

# Import the class from the module under test for patching if needed
from repobird_leadgen.github_search import GitHubSearcher
from repobird_leadgen.config import GITHUB_TOKEN, CACHE_DIR # Import CACHE_DIR for integration test

# Basic check to skip tests if no token is available
# Apply only to integration tests now
REASON_NO_TOKEN = "Skipping integration test: GITHUB_TOKEN not set in environment."
skip_if_no_token = pytest.mark.skipif(not GITHUB_TOKEN, reason=REASON_NO_TOKEN)

# --- Fixtures ---


@pytest.fixture
def mock_auth(mocker):
    """Mocks Auth.Token in the SUT's namespace."""
    mock_token_instance = MagicMock(spec=Auth.Token)
    # Patch Auth.Token in the SUT's namespace
    mock_auth_token_class = mocker.patch(
        "repobird_leadgen.github_search.Auth.Token", return_value=mock_token_instance
    )
    return {
        "class_patch": mock_auth_token_class,  # The mock object returned by patch
        "instance": mock_token_instance,  # The mock instance returned by the class patch
    }


@pytest.fixture
def mock_github(mocker):
    """Mocks the Github constructor and instance methods in the SUT's namespace.
       DOES NOT mock _execute_with_retry globally.
    """
    mock_instance = MagicMock(spec=Github)
    mock_search_results = MagicMock(spec=PaginatedList)
    # Make the mock iterable
    mock_search_results.__iter__.return_value = iter([])  # Default empty
    mock_instance.search_repositories.return_value = mock_search_results
    # Add __name__ attribute for retry logic logging
    mock_instance.search_repositories.__name__ = "search_repositories"


    # Mock the RateLimit object structure expected by the SUT
    mock_rate_limit_info = MagicMock()
    mock_rate_limit_info.search = MagicMock()
    mock_rate_limit_info.core = MagicMock() # Add mock for core limit
    # Set a realistic reset time slightly in the future
    reset_time = datetime.now(timezone.utc) + timedelta(seconds=10)
    mock_rate_limit_info.search.reset = reset_time
    mock_rate_limit_info.search.remaining = 50 # Add remaining calls
    mock_rate_limit_info.search.limit = 100 # Add limit
    mock_rate_limit_info.core.reset = reset_time
    mock_rate_limit_info.core.remaining = 500 # Add remaining calls
    mock_rate_limit_info.core.limit = 1000 # Add limit
    mock_instance.get_rate_limit.return_value = mock_rate_limit_info

    # Mock search_issues used by _has_open_issue_with_label
    mock_issue_search_results = MagicMock(spec=PaginatedList)
    mock_issue_search_results.totalCount = 0 # Default to no issues found
    mock_instance.search_issues.return_value = mock_issue_search_results
    # Add __name__ attribute for retry logic logging
    mock_instance.search_issues.__name__ = "search_issues"


    # Mock the Github constructor *within the SUT's namespace* ONLY
    mock_constructor_patch = mocker.patch(
        "repobird_leadgen.github_search.Github", return_value=mock_instance
    )

    return {
        "constructor_patch": mock_constructor_patch,  # The mock object returned by patch
        "instance": mock_instance,  # The mock instance returned by the constructor patch
        "search_results": mock_search_results,  # The mock PaginatedList returned by search_repositories
        "rate_limit_info": mock_rate_limit_info,  # The mock rate limit structure
        "issue_search_results": mock_issue_search_results, # Mock for issue search
    }


@pytest.fixture
def mock_repo_data():
    """Provides data for a mock Repository object."""
    now = datetime.now(timezone.utc)
    repo_name = "test_owner/test_repo"
    repo_url = f"https://github.com/{repo_name}"
    return {
        "full_name": repo_name,
        "html_url": repo_url,
        "description": "A test repository",
        "stargazers_count": 100,
        "forks_count": 20,
        "topics": ["python", "test"],
        "pushed_at": now,
        "owner": MagicMock(login="test_owner", type="User"),
        "readme_content": b"This is the README content.",
        # Ensure raw_data contains at least what's needed for caching
        "raw_data": {"full_name": repo_name, "html_url": repo_url, "description": "A test repository", "stargazers_count": 100}
    }


@pytest.fixture
def mock_repository(mocker, mock_repo_data):
    """Creates a mock Repository object."""
    mock_repo = MagicMock(spec=Repository)
    for key, value in mock_repo_data.items():
        if key != "readme_content": # Set all attributes except readme_content which needs separate handling
              setattr(mock_repo, key, value)

    # Mock get_readme if needed by any test (though search doesn't use it now)
    mock_readme_file = MagicMock(spec=ContentFile)
    mock_readme_file.decoded_content = mock_repo_data["readme_content"]
    mock_repo.get_readme.return_value = mock_readme_file

    return mock_repo

@pytest.fixture
def mock_searcher_env(mocker):
    """Mocks dependencies commonly used by GitHubSearcher methods."""
    mock_exists = mocker.patch("pathlib.Path.exists", return_value=False)
    # Mock pathlib.Path.open
    mock_path_open_patch = mocker.patch("pathlib.Path.open", mocker.mock_open())
    mock_mkdir = mocker.patch("pathlib.Path.mkdir")
    mock_sleep = mocker.patch("repobird_leadgen.github_search.time.sleep")
    mock_tqdm_class = mocker.patch("repobird_leadgen.github_search.tqdm")
    mock_pbar = MagicMock(name="mock_pbar_instance")
    mock_pbar.update = MagicMock(name="mock_pbar_update")
    mock_pbar.close = MagicMock(name="mock_pbar_close")
    mock_pbar.set_postfix_str = MagicMock(name="mock_pbar_set_postfix_str")
    mock_tqdm_instance = mock_tqdm_class.return_value
    mock_tqdm_instance.__enter__.return_value = mock_pbar
    mock_tqdm_instance.__exit__.return_value = None
    mock_print = mocker.patch("builtins.print")
    mock_json_dump = mocker.patch("repobird_leadgen.github_search.json.dump")
    # Mock _wait_for_rate_limit_reset with autospec=True
    mock_wait = mocker.patch("repobird_leadgen.github_search.GitHubSearcher._wait_for_rate_limit_reset", autospec=True)
    # Add __name__ to the mock itself for logging within SUT
    mock_wait.__name__ = "_wait_for_rate_limit_reset"


    return {
        "mock_exists": mock_exists,
        "mock_path_open": mock_path_open_patch, # Return the patch object for pathlib.Path.open
        "mock_mkdir": mock_mkdir,
        "mock_sleep": mock_sleep,
        "mock_tqdm_class": mock_tqdm_class,
        "mock_tqdm_pbar": mock_pbar,
        "mock_print": mock_print,
        "mock_json_dump": mock_json_dump,
        "mock_wait": mock_wait,
    }

# --- Unit Tests ---


@pytest.mark.unit
def test_github_searcher_init(mock_auth, mock_github, mock_searcher_env): # Added mock_searcher_env
    """Tests that GitHubSearcher initializes Auth.Token and Github correctly."""
    token = "test_token_123"
    searcher = GitHubSearcher(token=token)  # This call uses the patched constructors

    # Assert the Auth.Token mock (the patched class itself) was called
    mock_auth["class_patch"].assert_called_once_with(token)

    # Assert the Github mock (the patched class itself) was called
    mock_github["constructor_patch"].assert_called_once_with(
        auth=mock_auth["instance"], per_page=100, retry=5, timeout=15
    )
    # Assert the instance on the searcher is the one returned by the Github patch
    assert searcher.gh is mock_github["instance"]
    # Check if cache loading was attempted (mocked to not exist)
    mock_searcher_env["mock_print"].assert_any_call("[GitHubSearcher] Initialized. Using Browser Checker: False. Issue cache loaded with 0 entries.")


@pytest.mark.unit
def test_build_repo_query(mocker, mock_auth, mock_github, mock_searcher_env): # Added mock_searcher_env
    """Tests the _build_repo_query method with various parameters."""
    fixed_now = datetime(2024, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
    # Patch datetime in the SUT's namespace
    mock_datetime = mocker.patch("repobird_leadgen.github_search.datetime")
    mock_datetime.now.return_value = fixed_now
    # Allow datetime constructor usage if needed by SUT or tests
    mock_datetime.side_effect = (
        lambda *args, **kw: datetime(*args, **kw) if args else fixed_now
    )

    searcher = GitHubSearcher(token="dummy")  # Instantiation uses mocks

    # Test Case 1
    query1 = searcher._build_repo_query(
        language="python", min_stars=50, recent_days=30
    )
    # Corrected expected date format
    expected_date_30 = (fixed_now - timedelta(days=30)).strftime("%Y-%m-%d")
    assert "language:python" in query1
    assert "stars:>=50" in query1
    assert f"pushed:>{expected_date_30}" in query1
    assert "archived:false" in query1
    assert "fork:false" in query1
    assert "label:" not in query1 # Ensure label is NOT included

    # Test Case 2
    query2 = searcher._build_repo_query(
        language="javascript", min_stars=10, recent_days=90
    )
    # Corrected expected date format
    expected_date_90 = (fixed_now - timedelta(days=90)).strftime("%Y-%m-%d")
    assert "language:javascript" in query2
    assert "stars:>=10" in query2
    assert f"pushed:>{expected_date_90}" in query2
    assert "label:" not in query2

    # Test Case 3
    query3 = searcher._build_repo_query(
        language="go", min_stars=0, recent_days=1
    )
    # Corrected expected date format
    expected_date_1 = (fixed_now - timedelta(days=1)).strftime("%Y-%m-%d")
    assert "language:go" in query3
    assert "stars:>=0" in query3
    assert f"pushed:>{expected_date_1}" in query3
    assert "label:" not in query3


@pytest.mark.unit
# This test assumes the underlying API calls succeed, so retry logic shouldn't be hit.
# It relies on mock_github fixture where API calls are mocked, and mock_searcher_env where wait/sleep are mocked.
def test_search_success(
    mock_auth, mock_github, mock_repository, mock_repo_data, mock_searcher_env, tmp_path # Use common env mock
):
    searcher = GitHubSearcher(token="dummy")
    # searcher.gh is already the correct mock instance from the mock_github fixture

    mock_repos_list = [mock_repository]
    # Ensure the search_results mock is iterable
    mock_github["search_results"].__iter__.return_value = iter(mock_repos_list)
    mock_github["search_results"].totalCount = 1
    cache_file_path = tmp_path / "test_cache.jsonl"

    # Mock the issue check to return True for the target label
    mock_github["issue_search_results"].totalCount = 1

    label, lang, stars, days, max_res = "good first issue", "python", 50, 30, 10
    # Updated search call + list conversion
    results = list(searcher.search(
        label=label,
        language=lang,
        min_stars=stars,
        recent_days=days,
        max_results=max_res,
        cache_file=cache_file_path,
        existing_repo_names=set()
    ))

    # --- Assertions ---
    # 1. Check initial repo search call (should be called directly as no error/rate limit mocked)
    expected_repo_query = searcher._build_repo_query( # Use correct method
        language=lang, min_stars=stars, recent_days=days
    )
    mock_github["instance"].search_repositories.assert_called_once_with(
        query=expected_repo_query, sort="updated", order="desc"
    )


    # 2. Check issue search call for the specific label (should be called directly)
    expected_issue_query = f'repo:{mock_repository.full_name} is:issue is:open label:"{label}"' # Label has space
    mock_github["instance"].search_issues.assert_called_once_with(query=expected_issue_query)

    # 3. Check tqdm call
    mock_searcher_env["mock_tqdm_class"].assert_called_once()
    call_args, call_kwargs = mock_searcher_env["mock_tqdm_class"].call_args
    assert call_kwargs == {"total": max_res, "desc": f"Finding repos w/ '{label}' issues"}

    # 4. Check pbar updates
    mock_searcher_env["mock_tqdm_pbar"].update.assert_called_once_with(1)
    mock_searcher_env["mock_tqdm_pbar"].set_postfix_str.assert_any_call(f"Checking {mock_repository.full_name}...", refresh=True)
    mock_searcher_env["mock_tqdm_pbar"].set_postfix_str.assert_any_call("Found qualified", refresh=False)


    # 5. Check cache writing (Assert on the patch object for pathlib.Path.open)
    mock_searcher_env["mock_path_open"].assert_any_call(cache_file_path, "a", encoding="utf-8")
    handle = mock_searcher_env["mock_path_open"].return_value # Get the handle from the patch return value
    mock_searcher_env["mock_json_dump"].assert_any_call(mock_repository.raw_data, handle)
    handle.write.assert_any_call("\n")
    handle.flush.assert_called()


    # 6. Check final result
    assert len(results) == 1
    assert results[0].full_name == mock_repository.full_name
    # Check that retry helpers were NOT called
    mock_searcher_env["mock_sleep"].assert_not_called()
    mock_searcher_env["mock_wait"].assert_not_called()


@pytest.mark.unit
# This test assumes the underlying API calls succeed, so retry logic shouldn't be hit.
def test_search_repo_returned_even_if_readme_fails( # Renamed for clarity
    mock_auth,
    mock_github,
    mock_repository,
    mock_repo_data,
    mock_searcher_env, # Use common env mock
    tmp_path
):
    searcher = GitHubSearcher(token="dummy")

    # Mock the issue check to return True
    mock_github["issue_search_results"].totalCount = 1
    label="test"

    # Simulate README not found (though search doesn't fetch it anymore)
    mock_repository.get_readme.side_effect = UnknownObjectException(
        status=404, data={}, headers={}
    )
    mock_repos_list = [mock_repository]
    mock_github["search_results"].__iter__.return_value = iter(mock_repos_list)
    mock_github["search_results"].totalCount = 1
    cache_file_path = tmp_path / "test_cache_readme.jsonl"

    results = list(searcher.search(
        label=label, language="any", min_stars=0, recent_days=365,
        cache_file=cache_file_path,
        existing_repo_names=set()
    ))

    # Check repo search called directly
    mock_github["instance"].search_repositories.assert_called_once()

    # Check issue search called directly
    expected_issue_query = f"repo:{mock_repository.full_name} is:issue is:open label:{label}"
    mock_github["instance"].search_issues.assert_called_once_with(query=expected_issue_query)


    # Check pbar update was called as it qualifies based on label check
    mock_searcher_env["mock_tqdm_pbar"].update.assert_called_once_with(1)

    assert len(results) == 1  # Repo still qualifies based on issue label check
    assert results[0].full_name == mock_repository.full_name
    # Check retry helpers not called
    mock_searcher_env["mock_sleep"].assert_not_called()
    mock_searcher_env["mock_wait"].assert_not_called()
    # Assert get_readme was *not* called by the search function itself
    mock_repository.get_readme.assert_not_called()
    # Check cache writing (Assert on the patch object for pathlib.Path.open)
    mock_searcher_env["mock_path_open"].assert_any_call(cache_file_path, "a", encoding="utf-8")


@pytest.mark.unit
# Test the ACTUAL retry logic for rate limits during issue check.
# Relies on mock_searcher_env which mocks _wait_for_rate_limit_reset.
def test_search_rate_limit_during_issue_check( # Renamed for clarity
    mock_auth, mock_github, mock_repository, mock_repo_data, mock_searcher_env, tmp_path
):
    searcher = GitHubSearcher(token="dummy")
    # mock_wait is already active via mock_searcher_env

    repo1 = mock_repository
    # Make a second distinct mock repo
    repo2_data = mock_repo_data.copy()
    repo2_data["full_name"] = "owner/repo2"
    repo2_data["html_url"] = "https://github.com/owner/repo2"
    repo2_data["raw_data"] = {"full_name": "owner/repo2", "html_url": "https://github.com/owner/repo2"} # Add raw_data
    repo2 = MagicMock(spec=Repository)
    for key, value in repo2_data.items():
        setattr(repo2, key, value)


    rate_limit_exception = RateLimitExceededException(status=403, data={}, headers={"Retry-After": "1"}) # Add Retry-After header
    label = "ratelimit"
    cache_file_path = tmp_path / "test_cache_rate_limit.jsonl"

    # Mock issue search directly on the Github instance mock: Success for repo1, rate limit on repo2, success on retry for repo2
    issue_call_count = 0
    repo2_attempts = 0
    def mock_search_issues_rate_limit(*args, **kwargs):
        nonlocal issue_call_count, repo2_attempts
        issue_call_count += 1
        query = kwargs.get("query", "")
        if repo1.full_name in query:
            mock_res = MagicMock(spec=PaginatedList)
            mock_res.totalCount = 1 # repo1 qualifies
            return mock_res
        elif repo2.full_name in query:
            repo2_attempts += 1
            if repo2_attempts == 1: # First attempt for repo2 fails
                 raise rate_limit_exception
            else: # Second attempt for repo2 succeeds
                 mock_res = MagicMock(spec=PaginatedList)
                 mock_res.totalCount = 1 # repo2 also qualifies on retry
                 return mock_res
        else: # Should not happen in this test
            mock_res = MagicMock(spec=PaginatedList); mock_res.totalCount = 0; return mock_res


    mock_github["instance"].search_issues.side_effect = mock_search_issues_rate_limit

    # Configure repo search results
    mock_github["search_results"].__iter__.return_value = iter([repo1, repo2])
    mock_github["search_results"].totalCount = 2

    results = list(searcher.search(
        label=label, language="any", min_stars=0, recent_days=365, max_results=5,
        cache_file=cache_file_path,
        existing_repo_names=set()
    ))

    # --- Assertions ---
    # 1. Repo search called once
    mock_github["instance"].search_repositories.assert_called_once()

    # 2. search_issues called for repo1 (1 time) and repo2 (2 times due to retry)
    assert issue_call_count == 3
    expected_issue_query1 = f"repo:{repo1.full_name} is:issue is:open label:{label}"
    expected_issue_query2 = f"repo:{repo2.full_name} is:issue is:open label:{label}"
    mock_github["instance"].search_issues.assert_any_call(query=expected_issue_query1)
    mock_github["instance"].search_issues.assert_any_call(query=expected_issue_query2)


    # 3. _wait_for_rate_limit_reset (mocked by mock_searcher_env) called once by retry logic
    # Use ANY for the exception object as exact match might be tricky
    mock_searcher_env["mock_wait"].assert_called_once_with(searcher, "core", exception=ANY)

    # 4. TQDM update called twice (once for repo1, once for repo2 after retry)
    assert mock_searcher_env["mock_tqdm_pbar"].update.call_count == 2

    # 5. Both repos should be in results
    assert len(results) == 2
    assert results[0].full_name == repo1.full_name
    assert results[1].full_name == repo2.full_name

    # 6. Check print output for rate limit message from retry logic
    mock_searcher_env["mock_print"].assert_any_call("\nRate limit hit (attempt 1/5). Waiting...")

    # 7. Check cache writing (Assert on the patch object for pathlib.Path.open) - called twice
    assert mock_searcher_env["mock_path_open"].call_count == 1 # Only opened once for append
    mock_searcher_env["mock_path_open"].assert_any_call(cache_file_path, "a", encoding="utf-8")
    handle = mock_searcher_env["mock_path_open"].return_value
    assert mock_searcher_env["mock_json_dump"].call_count == 2 # Dumped twice
    mock_searcher_env["mock_json_dump"].assert_any_call(repo1.raw_data, handle)
    mock_searcher_env["mock_json_dump"].assert_any_call(repo2.raw_data, handle)
    assert handle.write.call_count == 2
    assert handle.flush.call_count > 0 # Called at least once


@pytest.mark.unit
# Test the ACTUAL retry logic for rate limits on initial search.
# Relies on mock_searcher_env which mocks _wait_for_rate_limit_reset.
def test_search_rate_limit_on_initial_search(mock_auth, mock_github, mock_searcher_env, tmp_path):
    searcher = GitHubSearcher(token="dummy")
    # mock_wait active via mock_searcher_env
    cache_file_path = tmp_path / "test_cache_initial_rate_limit.jsonl"

    # Configure the initial repo search to raise RateLimitExceededException persistently
    rate_limit_exception = RateLimitExceededException(status=403, data={}, headers={})
    mock_github["instance"].search_repositories.side_effect = rate_limit_exception

    # Expect the real _execute_with_retry to raise RuntimeError after failing all retries
    with pytest.raises(RuntimeError, match="Failed to execute search_repositories after 5 retries"):
        list(searcher.search(
            label="test", language="any", min_stars=0, recent_days=365,
            cache_file=cache_file_path,
            existing_repo_names=set()
        ))

    # Check that search_repositories was called multiple times (5 retries)
    assert mock_github["instance"].search_repositories.call_count == 5

    # Check that _wait_for_rate_limit_reset (mocked) was called 5 times (once per retry)
    assert mock_searcher_env["mock_wait"].call_count == 5
    # Use ANY for the exception object
    mock_searcher_env["mock_wait"].assert_called_with(searcher, "search", exception=ANY) # Check last call args

    # TQDM should not be reached
    mock_searcher_env["mock_tqdm_class"].assert_not_called()
    # File should not be opened if initial search fails
    mock_searcher_env["mock_path_open"].assert_not_called()


@pytest.mark.unit
# Test the ACTUAL retry logic for generic exceptions during issue check.
# Relies on mock_searcher_env which mocks time.sleep.
def test_search_generic_github_exception_during_issue_check( # Renamed for clarity
    mock_auth, mock_github, mock_repository, mock_repo_data, mock_searcher_env, tmp_path
):
    searcher = GitHubSearcher(token="dummy")
    # mock_sleep active via mock_searcher_env

    repo1 = mock_repository
    # Make a second distinct mock repo
    repo2_data = mock_repo_data.copy()
    repo2_data["full_name"] = "owner/repo2"
    repo2_data["html_url"] = "https://github.com/owner/repo2"
    repo2_data["raw_data"] = {"full_name": "owner/repo2", "html_url": "https://github.com/owner/repo2"}
    repo2 = MagicMock(spec=Repository)
    for key, value in repo2_data.items():
        setattr(repo2, key, value)

    generic_exception = GithubException(status=500, data={"message": "Server Error"}, headers={}) # Simulate server error
    label = "genericerror"
    cache_file_path = tmp_path / "test_cache_generic_exception.jsonl"

    # Mock issue search directly on Github instance mock: Success for repo1, generic error on repo2 (persistent)
    issue_call_count = 0
    repo2_attempts = 0
    def mock_search_issues_generic_error(*args, **kwargs):
        nonlocal issue_call_count, repo2_attempts
        issue_call_count += 1
        query = kwargs.get("query", "")
        if repo1.full_name in query:
            mock_res = MagicMock(spec=PaginatedList)
            mock_res.totalCount = 1 # repo1 qualifies
            return mock_res
        elif repo2.full_name in query:
             repo2_attempts += 1
             # Raise the error consistently for repo2
             raise generic_exception
        else: # Should not happen
             mock_res = MagicMock(spec=PaginatedList); mock_res.totalCount = 0; return mock_res

    mock_github["instance"].search_issues.side_effect = mock_search_issues_generic_error

    # Configure repo search results
    mock_github["search_results"].__iter__.return_value = iter([repo1, repo2])
    mock_github["search_results"].totalCount = 2

    results = list(searcher.search(
        label=label, language="any", min_stars=0, recent_days=365,
        cache_file=cache_file_path,
        existing_repo_names=set()
    ))

    # --- Assertions ---
    # 1. Repo search called once
    mock_github["instance"].search_repositories.assert_called_once()

    # 2. search_issues called for repo1 (1 time) and repo2 (multiple times via REAL retry)
    # Total calls = 1 (repo1) + 6 (repo2 initial + 5 retries) = 7 times.
    assert issue_call_count == 7 # This assertion might still fail if retry logic is broken for generic exceptions
    expected_issue_query1 = f"repo:{repo1.full_name} is:issue is:open label:{label}"
    expected_issue_query2 = f"repo:{repo2.full_name} is:issue is:open label:{label}"
    # Check calls
    calls_to_search_issues = mock_github["instance"].search_issues.call_args_list
    repo1_calls = [c for c in calls_to_search_issues if repo1.full_name in c.kwargs.get("query", "")]
    repo2_calls = [c for c in calls_to_search_issues if repo2.full_name in c.kwargs.get("query", "")]
    assert len(repo1_calls) == 1
    assert len(repo2_calls) == 6 # Initial + 5 retries


    # 3. Sleep (mocked by mock_searcher_env) called multiple times by retry logic's backoff
    assert mock_searcher_env["mock_sleep"].call_count == 5 # Expect 5 calls for 5 retries

    # 4. TQDM update called once for the qualified repo (repo1)
    mock_searcher_env["mock_tqdm_pbar"].update.assert_called_once_with(1)

    # 5. Only repo1 should be in results as repo2 check failed permanently
    assert len(results) == 1
    assert results[0].full_name == repo1.full_name

    # 6. Check print output for the error message after retries fail
    expected_error_msg_part = "Failed to execute _has_open_issue_with_label after 5 retries" # Match part of the message
    # Check *any* call contains the expected pattern
    found_print_error = False
    for call_args, call_kwargs in mock_searcher_env["mock_print"].call_args_list:
        if call_args and isinstance(call_args[0], str) and expected_error_msg_part in call_args[0]:
             # Check that the repo name is also present in the error message
             if repo2.full_name in call_args[0]:
                 found_print_error = True
                 break
    assert found_print_error, f"Expected print call containing '{expected_error_msg_part}' and '{repo2.full_name}' not found"


    # Check retry messages from _execute_with_retry
    mock_searcher_env["mock_print"].assert_any_call(f"\nGitHub API server error (Status 500, attempt 1/5): Server Error. Retrying in {mock_searcher_env['mock_sleep'].call_args_list[0][0][0]:.2f}s...")

    # 7. Check cache writing (Assert on the patch object for pathlib.Path.open) - called once for repo1
    assert mock_searcher_env["mock_path_open"].call_count == 1 # Opened once
    mock_searcher_env["mock_path_open"].assert_any_call(cache_file_path, "a", encoding="utf-8")
    handle = mock_searcher_env["mock_path_open"].return_value
    assert mock_searcher_env["mock_json_dump"].call_count == 1 # Dumped once
    mock_searcher_env["mock_json_dump"].assert_any_call(repo1.raw_data, handle)
    assert handle.write.call_count == 1
    assert handle.flush.call_count > 0


# --- Integration Tests ---


@pytest.mark.integration
@skip_if_no_token
def test_basic_search_integration(tmp_path): # Added tmp_path
    # Ensure CACHE_DIR exists for the integration test's issue cache
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    gh = GitHubSearcher() # Uses real token and CACHE_DIR
    cache_file_path = tmp_path / "integration_test_cache.jsonl" # Main repo cache
    label = "good first issue" # Define label
    max_results_integration = 3 # Keep low for testing speed
    try:
        # This uses the REAL _execute_with_retry
        repos = list(gh.search(
            label=label, # Use defined label
            language="python",
            min_stars=1, # Low stars for finding results
            recent_days=365 * 3, # Wider range
            max_results=max_results_integration, # Keep low
            cache_file=cache_file_path,
            existing_repo_names=set()
        ))
        # Check type after list conversion
        assert isinstance(repos, list)
        # Allow empty list if no results found matching criteria
        if repos:
            print(f"\nIntegration test found {len(repos)} qualifying repos.")
            # Check the type of elements within the list
            assert all(isinstance(r, Repository) for r in repos)
            # Check we didn't exceed max_results
            assert len(repos) <= max_results_integration
            # Check cache file was created and has content (if repos were found)
            assert cache_file_path.exists()
            assert cache_file_path.stat().st_size > 0

            # Optional: Check if issue cache was populated (use path from instance)
            issue_cache_path = Path(gh.issue_cache_path)
            if issue_cache_path.exists():
                 with issue_cache_path.open("r") as f:
                     lines = f.readlines()
                     print(f"Issue cache file found with {len(lines)} entries.")
                     # This might be 0 if all repos checked were already in cache
                     # assert len(lines) > 0
            else:
                 print("Issue cache file not found after integration test.")


        else:
            print("\nIntegration test found 0 qualifying repos. Main repo cache file might be empty.")
            # If no repos found, cache file might be empty or not created if loop didn't run
            # assert not cache_file_path.exists() or cache_file_path.stat().st_size == 0

    except RateLimitExceededException:
        pytest.skip("GitHub API rate limit exceeded during integration test.")
    except GithubException as e:
        pytest.fail(f"GitHub API error during test_basic_search: {e.status} {e.data}")
    except Exception as e:
        pytest.fail(f"Unexpected error during test_basic_search: {e}")

# Removed test_good_first_issue_repos_helper_integration
