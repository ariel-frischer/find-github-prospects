import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone
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
import copy # Import copy module

# Import the class from the module under test for patching if needed
from repobird_leadgen.github_search import GitHubSearcher
from repobird_leadgen.config import GITHUB_TOKEN  # Keep for integration tests

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
    """Mocks the Github constructor in the SUT's namespace and provides the instance."""
    mock_instance = MagicMock(spec=Github)
    mock_search_results = MagicMock(spec=PaginatedList)
    # Make the mock iterable
    mock_search_results.__iter__.return_value = iter([])  # Default empty

    # Create mock method and set __name__ explicitly for retry logic
    mock_search_method = MagicMock(name='search_repositories', return_value=mock_search_results)
    mock_search_method.__name__ = 'search_repositories'
    mock_instance.search_repositories = mock_search_method

    # Mock the RateLimit object structure expected by the SUT
    mock_rate_limit_info = MagicMock()
    mock_rate_limit_info.search = MagicMock()
    mock_rate_limit_info.core = MagicMock() # Add core limit mock
    # Set a realistic reset time slightly in the future
    reset_time = datetime.now(timezone.utc) + timedelta(seconds=10)
    mock_rate_limit_info.search.reset = reset_time
    mock_rate_limit_info.core.reset = reset_time
    mock_instance.get_rate_limit.return_value = mock_rate_limit_info

    # Mock the Github constructor *within the SUT's namespace* ONLY
    mock_constructor_patch = mocker.patch(
        "repobird_leadgen.github_search.Github", return_value=mock_instance
    )

    return {
        "constructor_patch": mock_constructor_patch,  # The mock object returned by patch
        "instance": mock_instance,  # The mock instance returned by the constructor patch
        "search_results": mock_search_results,  # The mock PaginatedList returned by search_repositories
        "rate_limit_info": mock_rate_limit_info,  # The mock rate limit structure
    }


@pytest.fixture
def mock_repo_data():
    """Provides data for a mock Repository object."""
    now = datetime.now(timezone.utc)
    return {
        "full_name": "test_owner/test_repo",
        "html_url": "https://github.com/test_owner/test_repo",
        "description": "A test repository",
        "stargazers_count": 100,
        "forks_count": 20,
        "topics": ["python", "test"],
        "pushed_at": now,
        "owner": MagicMock(login="test_owner", type="User"),
        "readme_content": b"This is the README content.",
        "raw_data": { # Add raw_data for caching
            "full_name": "test_owner/test_repo",
            "html_url": "https://github.com/test_owner/test_repo",
            "description": "A test repository",
            "stargazers_count": 100,
            "forks_count": 20,
            "topics": ["python", "test"],
            "pushed_at": now.isoformat(),
            "owner": {"login": "test_owner", "type": "User"},
        }
    }


@pytest.fixture
def mock_repository(mocker, mock_repo_data):
    """Creates a mock Repository object."""
    mock_repo = MagicMock(spec=Repository)
    for key, value in mock_repo_data.items():
        if key != "readme_content":
            setattr(mock_repo, key, value)

    mock_readme_file = MagicMock(spec=ContentFile)
    mock_readme_file.decoded_content = mock_repo_data["readme_content"]
    mock_repo.get_readme.return_value = mock_readme_file

    # Mock the _has_open_issue_with_label method on the searcher instance later
    # We also need to mock the raw_data attribute for caching
    # Use deepcopy to avoid modifying the original fixture data if reused
    mock_repo.raw_data = copy.deepcopy(mock_repo_data["raw_data"])

    return mock_repo


# --- Unit Tests ---


@pytest.mark.unit
def test_github_searcher_init(mock_auth, mock_github):
    """Tests that GitHubSearcher initializes Auth.Token and Github correctly."""
    token = "test_token_123"
    searcher = GitHubSearcher(token=token)
    # Assert the Auth.Token mock (the patched class itself) was called
    mock_auth["class_patch"].assert_called_once_with(token)

    # Assert the Github mock (the patched class itself) was called
    mock_github["constructor_patch"].assert_called_once_with(
        auth=mock_auth["instance"], per_page=100, retry=5, timeout=15
    )
    # Assert the instance on the searcher is the one returned by the Github patch
    assert searcher.gh is mock_github["instance"]


@pytest.mark.unit
@patch("repobird_leadgen.github_search.datetime") # Patch datetime directly in SUT namespace
def test_build_repo_query(mock_datetime, mock_auth, mock_github):  # Need fixtures for init
    """Tests the _build_repo_query method with various parameters."""
    fixed_now = datetime(2024, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = fixed_now
    # Allow datetime constructor usage if needed by SUT or tests
    mock_datetime.side_effect = (
        lambda *args, **kw: datetime(*args, **kw) if args else fixed_now
    )

    searcher = GitHubSearcher(token="dummy")

    # Call _build_repo_query without 'label'
    query1 = searcher._build_repo_query(
        language="python", min_stars=50, recent_days=30
    )
    expected_date_30_str = (fixed_now - timedelta(days=30)).strftime("%Y-%m-%d")
    assert f"language:python" in query1
    assert f"stars:>=50" in query1
    assert f"pushed:>{expected_date_30_str}" in query1
    assert "label:" not in query1 # Ensure label is not part of this query

    query2 = searcher._build_repo_query(
        language="javascript", min_stars=10, recent_days=90
    )
    expected_date_90_str = (fixed_now - timedelta(days=90)).strftime("%Y-%m-%d")
    assert f"language:javascript" in query2
    assert f"stars:>=10" in query2
    assert f"pushed:>{expected_date_90_str}" in query2
    assert "label:" not in query2

    query3 = searcher._build_repo_query(
        language="go", min_stars=0, recent_days=1
    )
    expected_date_1_str = (fixed_now - timedelta(days=1)).strftime("%Y-%m-%d")
    assert f"language:go" in query3
    assert f"stars:>=0" in query3
    assert f"pushed:>{expected_date_1_str}" in query3
    assert "label:" not in query3


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm")  # Patch tqdm directly
@patch("repobird_leadgen.github_search.time.sleep")
@patch.object(GitHubSearcher, "_has_open_issue_with_label", return_value=True) # Mock label check
@patch("repobird_leadgen.github_search.datetime") # Patch datetime for query build
def test_search_success(
    mock_datetime, mock_label_check, mock_sleep, mock_tqdm_class, mock_auth, mock_github, mock_repository, mock_repo_data, tmp_path # Added tmp_path
):
    # Setup datetime mock for _build_repo_query
    fixed_now = datetime(2024, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = fixed_now
    mock_datetime.side_effect = (
        lambda *args, **kw: datetime(*args, **kw) if args else fixed_now
    )

    # Configure the mock tqdm class to handle context management
    mock_pbar = MagicMock()
    mock_pbar.update = MagicMock()
    mock_pbar.close = MagicMock()
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_context.__exit__.return_value = None
    mock_tqdm_class.return_value = (
        mock_tqdm_context  # Calling tqdm() returns the context manager
    )

    searcher = GitHubSearcher(token="dummy")
    searcher.issue_cache = {} # Clear cache for this test
    # searcher.gh is already the correct mock instance from the mock_github fixture

    mock_repos_list = [mock_repository]
    # Ensure the search_results mock is iterable
    mock_github["search_results"].__iter__.return_value = iter(mock_repos_list)
    mock_github["search_results"].totalCount = 1

    label, lang, stars, days, max_res = "good first issue", "python", 50, 30, 10
    # Consume the iterator
    results = list(searcher.search(
        label=label,
        language=lang,
        min_stars=stars,
        recent_days=days,
        max_results=max_res,
        cache_file=tmp_path / "test_cache.json", # Added cache_file
        existing_repo_names=set() # Provide empty set
    ))

    # Build expected query without label
    expected_query = searcher._build_repo_query(
        language=lang, min_stars=stars, recent_days=days
    )
    # Assert the initial repo search was called correctly
    mock_github["instance"].search_repositories.assert_called_once_with(
        query=expected_query, sort="updated", order="desc"
    )
    # Assert the label check was called on the searcher instance via _execute_with_retry
    mock_label_check.assert_called_once_with(mock_repository, label)


    # Check tqdm call using call_args
    mock_tqdm_class.assert_called_once()
    call_args, call_kwargs = mock_tqdm_class.call_args
    # The first positional argument should be the iterable
    assert call_args == ()  # No positional args expected
    # Check the description and total (now max_results)
    assert call_kwargs["desc"] == f"Finding repos w/ '{label}' issues"
    assert call_kwargs["total"] == max_res

    # Check update on the pbar instance
    mock_pbar.update.assert_called_once_with(1)

    assert len(results) == 1
    assert results[0] == mock_repository
    mock_sleep.assert_not_called()


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm")
@patch("repobird_leadgen.github_search.time.sleep")
@patch("builtins.print")
@patch.object(GitHubSearcher, "_has_open_issue_with_label", return_value=False) # Mock label check returns False
@patch("repobird_leadgen.github_search.datetime") # Patch datetime for query build
def test_search_repo_without_label(
    mock_datetime, mock_label_check, mock_print, mock_sleep, mock_tqdm_class, mock_auth, mock_github, mock_repository, tmp_path
):
    """Test that a repo is NOT yielded if the label check returns False."""
    # Setup datetime mock for _build_repo_query
    fixed_now = datetime(2024, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = fixed_now
    mock_datetime.side_effect = (
        lambda *args, **kw: datetime(*args, **kw) if args else fixed_now
    )

    mock_pbar = MagicMock()
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_class.return_value = mock_tqdm_context

    searcher = GitHubSearcher(token="dummy")
    searcher.issue_cache = {} # Clear cache for this test
    mock_repos_list = [mock_repository]
    mock_github["search_results"].__iter__.return_value = iter(mock_repos_list)
    mock_github["search_results"].totalCount = 1

    label = "non-existent-label"
    # Consume iterator
    results = list(searcher.search(
        label=label, language="any", min_stars=0, recent_days=365, max_results=1,
        cache_file=tmp_path / "test_cache.json", existing_repo_names=set()
    ))

    mock_github["instance"].search_repositories.assert_called_once()
    # Assert the label check was called via _execute_with_retry
    mock_label_check.assert_called_once_with(mock_repository, label)
    mock_tqdm_class.assert_called_once()
    # pbar.update should NOT have been called because the repo didn't qualify
    mock_pbar.update.assert_not_called()
    assert len(results) == 0 # No results should be yielded
    mock_sleep.assert_not_called()


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm")
@patch("repobird_leadgen.github_search.time.sleep")
@patch("builtins.print")
@patch.object(GitHubSearcher, "_has_open_issue_with_label") # Mock label check
@patch("repobird_leadgen.github_search.datetime") # Patch datetime for query build
def test_search_label_check_exception(
    mock_datetime, mock_label_check, mock_print, mock_sleep, mock_tqdm_class, mock_auth, mock_github, mock_repository, tmp_path
):
    """Test that repo is skipped if label check raises an exception after retries."""
     # Setup datetime mock for _build_repo_query
    fixed_now = datetime(2024, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
    mock_datetime.now.return_value = fixed_now
    mock_datetime.side_effect = (
        lambda *args, **kw: datetime(*args, **kw) if args else fixed_now
    )

    mock_pbar = MagicMock()
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_class.return_value = mock_tqdm_context

    searcher = GitHubSearcher(token="dummy")
    searcher.issue_cache = {} # Clear cache for this test
    mock_repos_list = [mock_repository]
    mock_github["search_results"].__iter__.return_value = iter(mock_repos_list)
    mock_github["search_results"].totalCount = 1

    # Configure label check mock to raise a retryable exception
    mock_label_check.side_effect = GithubException(status=500, data={}, headers={})
    # Set __name__ on the mock for the retry logic error message
    mock_label_check.__name__ = '_has_open_issue_with_label'

    label = "problem-label"
    # Consume iterator
    results = list(searcher.search(
        label=label, language="any", min_stars=0, recent_days=365, max_results=1,
        cache_file=tmp_path / "test_cache.json", existing_repo_names=set()
    ))

    mock_github["instance"].search_repositories.assert_called_once()
    # Assert the label check was called with the correct arguments (at least once due to retry)
    mock_label_check.assert_any_call(mock_repository, label)
    # Assert it was called multiple times (specifically 5 times for max_retries)
    assert mock_label_check.call_count == 5
    # Assert sleep was called due to retries
    mock_sleep.assert_called()

    mock_tqdm_class.assert_called_once()
    mock_pbar.update.assert_not_called() # No update as repo is skipped after retries fail
    assert len(results) == 0 # No results yielded


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm")
@patch("repobird_leadgen.github_search.time.sleep")
@patch.object(GitHubSearcher, "_has_open_issue_with_label", return_value=True) # Mock label check
@patch("repobird_leadgen.github_search.datetime") # Patch datetime
def test_search_rate_limit_during_iteration(
    mock_datetime, mock_label_check, mock_sleep, mock_tqdm_class, mock_auth, mock_github, tmp_path # Removed mock_repository, mock_repo_data
):
    # Setup datetime mock for _build_repo_query and wait logic
    mock_now = datetime.now(timezone.utc)
    mock_datetime.now.return_value = mock_now
    mock_datetime.side_effect = (
        lambda *args, **kw: datetime(*args, **kw) if args else mock_now
    )

    # Configure the mock tqdm class
    mock_pbar = MagicMock()
    mock_pbar.update = MagicMock()
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_class.return_value = mock_tqdm_context

    searcher = GitHubSearcher(token="dummy")
    searcher.issue_cache = {} # Clear cache for this test

    # --- Manually define base data (similar to mock_repo_data fixture) ---
    base_repo_data = {
        "full_name": "test_owner/test_repo",
        "html_url": "https://github.com/test_owner/test_repo",
        "description": "A test repository",
        "stargazers_count": 100,
        "forks_count": 20,
        "topics": ["python", "test"],
        "pushed_at": mock_now, # Use mocked now
        "owner": MagicMock(login="test_owner", type="User"),
        # "readme_content": b"This is the README content.", # Not needed directly for mock
        "raw_data": {
            "full_name": "test_owner/test_repo",
            "html_url": "https://github.com/test_owner/test_repo",
            "description": "A test repository",
            "stargazers_count": 100,
            "forks_count": 20,
            "topics": ["python", "test"],
            "pushed_at": mock_now.isoformat(), # Use mocked now
            "owner": {"login": "test_owner", "type": "User"},
        }
    }
    # --- Create mocks based on this data ---
    repo1_data = copy.deepcopy(base_repo_data)
    repo1 = MagicMock(spec=Repository)
    for key, value in repo1_data.items():
         if key != "readme_content":
            setattr(repo1, key, value)
    repo1.raw_data = repo1_data["raw_data"]

    # Create data for the second repo based on the base data
    repo2_data = copy.deepcopy(base_repo_data)
    repo2_data["full_name"] = "owner/repo2"
    repo2_data["raw_data"]["full_name"] = "owner/repo2"
    # Create the second mock repository
    repo2 = MagicMock(spec=Repository)
    for key, value in repo2_data.items():
         if key != "readme_content":
            setattr(repo2, key, value)
    repo2.raw_data = repo2_data["raw_data"]

    rate_limit_exception = RateLimitExceededException(status=403, data={}, headers={})

    # Simulate iterator raising RateLimitExceededException after yielding the first repo
    def rate_limit_iterator():
        yield repo1
        raise rate_limit_exception

    mock_github["search_results"].__iter__.return_value = rate_limit_iterator()
    # totalCount should reflect the potential total if no error occurred
    mock_github["search_results"].totalCount = 2

    # Configure rate limit info mock for the wait logic (use core limit for iteration)
    mock_github["rate_limit_info"].core.reset = mock_now + timedelta(seconds=0.01)

    label = "test"
    # Consume iterator
    results = list(searcher.search(
        label=label, language="any", min_stars=0, recent_days=365, max_results=5,
        cache_file=tmp_path / "test_cache.json", existing_repo_names=set()
    ))

    # Assertions
    mock_github["instance"].search_repositories.assert_called_once()
    # Label check should only be called for the first repo before the exception
    mock_label_check.assert_called_once_with(repo1, label)
    # get_rate_limit should be called by _wait_for_rate_limit_reset when handling RLE from iterator
    mock_github["instance"].get_rate_limit.assert_called()
    mock_sleep.assert_called() # Sleep should be called due to rate limit wait

    mock_tqdm_class.assert_called_once()
    # pbar update called once for the successfully processed repo1
    mock_pbar.update.assert_called_once_with(1)

    assert len(results) == 1 # Only repo1 should be yielded
    assert results[0] == repo1


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm")  # Patch tqdm directly
@patch("repobird_leadgen.github_search.time.sleep")
@patch("repobird_leadgen.github_search.datetime") # Patch datetime
def test_search_rate_limit_on_initial_search(mock_datetime, mock_sleep, mock_tqdm_class, mock_auth, mock_github, tmp_path): # Added tmp_path
     # Setup datetime mock for wait logic
    mock_now = datetime.now(timezone.utc)
    mock_datetime.now.return_value = mock_now
    mock_datetime.side_effect = (
        lambda *args, **kw: datetime(*args, **kw) if args else mock_now
    )

    searcher = GitHubSearcher(token="dummy")
    searcher.issue_cache = {} # Clear cache for this test

    # Configure search_repositories to raise RateLimitExceededException
    rate_limit_exception = RateLimitExceededException(status=403, data={}, headers={})
    # Ensure the mock method has the correct name explicitly in the test
    mock_github["instance"].search_repositories.__name__ = 'search_repositories'
    mock_github["instance"].search_repositories.side_effect = rate_limit_exception

    # Configure rate limit info mock for the wait logic (use search limit for search_repositories)
    mock_github["rate_limit_info"].search.reset = mock_now + timedelta(seconds=0.01)

    # The search function now uses _execute_with_retry internally.
    # We expect _execute_with_retry to catch the RLE, wait, and retry.
    # If retries fail, it raises RuntimeError.
    with pytest.raises(RuntimeError, match="Failed to execute search_repositories after 5 retries"):
         # Consume the iterator to trigger the search and retries
        list(searcher.search(
            label="test", language="any", min_stars=0, recent_days=365,
            cache_file=tmp_path / "test_cache.json", existing_repo_names=set()
        ))

    # Assert search_repositories was called multiple times due to retries
    assert mock_github["instance"].search_repositories.call_count == 5
    # Assert get_rate_limit was called during the wait logic
    mock_github["instance"].get_rate_limit.assert_called()
    # Assert sleep was called during the wait logic
    mock_sleep.assert_called()
    mock_tqdm_class.assert_not_called()  # tqdm shouldn't be reached if initial search fails


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm")
@patch("repobird_leadgen.github_search.time.sleep")
@patch.object(GitHubSearcher, "_has_open_issue_with_label", return_value=True) # Mock label check
@patch("repobird_leadgen.github_search.datetime") # Patch datetime
def test_search_generic_github_exception_during_iteration(
    mock_datetime, mock_label_check, mock_sleep, mock_tqdm_class, mock_auth, mock_github, mock_repository, tmp_path # Added tmp_path
):
    # Setup datetime mock
    mock_now = datetime.now(timezone.utc)
    mock_datetime.now.return_value = mock_now
    mock_datetime.side_effect = (
        lambda *args, **kw: datetime(*args, **kw) if args else mock_now
    )

    # Configure the mock tqdm class
    mock_pbar = MagicMock(name="mock_pbar_instance")
    mock_pbar.update = MagicMock(name="mock_pbar_update")
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_class.return_value = mock_tqdm_context

    searcher = GitHubSearcher(token="dummy")
    searcher.issue_cache = {} # Clear cache for this test

    repo1 = mock_repository
    generic_exception = GithubException(status=500, data={}, headers={}) # Retryable error

    # Simulate iterator raising GithubException after yielding the first repo
    def generic_exception_iterator():
        yield repo1
        raise generic_exception

    mock_github["search_results"].__iter__.return_value = generic_exception_iterator()
    mock_github["search_results"].totalCount = 2

    # Consume iterator
    results = list(searcher.search(
        label="test", language="any", min_stars=0, recent_days=365, max_results=2,
        cache_file=tmp_path / "test_cache.json", existing_repo_names=set()
    ))

    # Assertions
    mock_github["instance"].search_repositories.assert_called_once()
    # Label check called only for repo1
    mock_label_check.assert_called_once_with(repo1, "test")
    # Sleep should NOT be called here because the exception occurs during iteration (getting next item),
    # which the current search loop catches and skips, rather than retrying the iteration itself.
    mock_sleep.assert_not_called()

    mock_tqdm_class.assert_called_once()
    # pbar update called once for repo1
    mock_pbar.update.assert_called_once_with(1)

    assert len(results) == 1 # Only repo1 yielded
    assert results[0] == repo1


# --- Integration Tests ---


@pytest.mark.integration
@skip_if_no_token
def test_basic_search_integration(tmp_path): # Added tmp_path
    gh = GitHubSearcher()
    cache_file = tmp_path / "integration_test_cache.json"
    try:
        # Consume the iterator and convert to list
        repos = list(gh.search(
            label="good first issue",
            language="python",
            min_stars=1,
            recent_days=365 * 2,
            max_results=1, # Reduced max_results to 1 for debugging timeout
            cache_file=cache_file, # Use tmp_path cache
            existing_repo_names=set() # Provide empty set
        ))
        assert isinstance(repos, list) # Check if it's a list after conversion
        if repos:
            assert isinstance(repos[0], Repository)
            # Verify cache file was created and has content
            assert cache_file.exists()
            assert cache_file.stat().st_size > 0
            # Very basic check for JSONL format (can be improved)
            with cache_file.open("r") as f:
                first_line = f.readline()
                assert first_line.startswith("{") and first_line.endswith("}\n")

        # The number of results yielded might be less than max_results
        assert len(repos) <= 1 # Adjusted assertion for max_results=1
    except RateLimitExceededException:
        pytest.skip("GitHub API rate limit exceeded.")
    except GithubException as e:
        pytest.fail(f"GitHub API error during test_basic_search: {e.status} {e.data}")
    except Exception as e:
        pytest.fail(f"Unexpected error during test_basic_search: {e}")

# Removed test_good_first_issue_repos_helper_integration function
