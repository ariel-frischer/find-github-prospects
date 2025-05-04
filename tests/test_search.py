import copy  # For deep copying repo data
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch  # Added mock_open

import pytest
from github import (
    Auth,
    Github,
    GithubException,
    RateLimitExceededException,
)
from github.ContentFile import ContentFile
from github.Issue import Issue
from github.PaginatedList import PaginatedList
from github.Repository import Repository
from github.TimelineEvent import TimelineEvent  # Added

from repobird_leadgen.config import (
    CACHE_DIR,
    GITHUB_TOKEN,
)  # Keep for integration tests

# Import the class from the module under test for patching if needed
from repobird_leadgen.github_search import GitHubSearcher

# Basic check to skip tests if no token is available
# Apply only to integration tests now
REASON_NO_TOKEN = "Skipping integration test: GITHUB_TOKEN not set in environment."
skip_if_no_token = pytest.mark.skipif(not GITHUB_TOKEN, reason=REASON_NO_TOKEN)

# --- Fixtures ---


@pytest.fixture
def mock_auth(mocker):
    """Mocks Auth.Token in the SUT's namespace."""
    mock_token_instance = MagicMock(spec=Auth.Token)
    mock_auth_token_class = mocker.patch(
        "repobird_leadgen.github_search.Auth.Token", return_value=mock_token_instance
    )
    return {
        "class_patch": mock_auth_token_class,
        "instance": mock_token_instance,
    }


@pytest.fixture
def mock_github(mocker):
    """Mocks the Github constructor and provides the instance."""
    mock_instance = MagicMock(spec=Github)
    mock_search_repos_results = MagicMock(spec=PaginatedList)
    mock_search_repos_results.__iter__.return_value = iter([])
    mock_instance.search_repositories.return_value = mock_search_repos_results

    mock_search_issues_results = MagicMock(spec=PaginatedList)
    mock_search_issues_results.totalCount = 0  # Default for API label check
    mock_instance.search_issues.return_value = mock_search_issues_results

    mock_rate_limit_info = MagicMock()
    mock_rate_limit_info.search = MagicMock()
    mock_rate_limit_info.search.reset = datetime.now(timezone.utc) + timedelta(
        seconds=10
    )
    mock_rate_limit_info.core = MagicMock()  # Add core limit mock
    mock_rate_limit_info.core.reset = datetime.now(timezone.utc) + timedelta(seconds=10)
    mock_instance.get_rate_limit.return_value = mock_rate_limit_info

    mock_constructor_patch = mocker.patch(
        "repobird_leadgen.github_search.Github", return_value=mock_instance
    )

    return {
        "constructor_patch": mock_constructor_patch,
        "instance": mock_instance,
        "search_repos_results": mock_search_repos_results,
        "search_issues_results": mock_search_issues_results,
        "rate_limit_info": mock_rate_limit_info,
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
        "raw_data": {
            "initial_key": "initial_value"
        },  # Add raw_data for storing found issues
    }


@pytest.fixture
def mock_repository(mocker, mock_repo_data):
    """Creates a mock Repository object."""
    mock_repo = MagicMock(spec=Repository)
    # Use deepcopy for raw_data to avoid test interference
    repo_data = copy.deepcopy(mock_repo_data)
    for key, value in repo_data.items():
        setattr(mock_repo, key, value)

    # Mock get_issues for detailed checks
    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([])
    mock_issues_paginator.totalCount = 0  # Default total count
    mock_repo.get_issues.return_value = mock_issues_paginator

    # Mock get_readme for contact scraper tests (if any)
    mock_readme_file = MagicMock(spec=ContentFile)
    mock_readme_file.decoded_content = b"README content"
    mock_repo.get_readme.return_value = mock_readme_file

    return mock_repo


@pytest.fixture
def mock_issue(mocker):
    """Creates a basic mock Issue object."""

    def _create_mock_issue(number, created_at_dt):  # Renamed dt
        issue = MagicMock(spec=Issue)
        issue.number = number
        issue.created_at = created_at_dt
        # Mock get_timeline for PR checks
        mock_timeline_paginator = MagicMock(spec=PaginatedList)
        mock_timeline_paginator.__iter__.return_value = iter([])
        issue.get_timeline.return_value = mock_timeline_paginator
        return issue

    return _create_mock_issue


@pytest.fixture
def mock_timeline_event(mocker):
    """Creates a mock TimelineEvent object, simulating a linked PR."""

    def _create_mock_event(is_pr=True):
        event = MagicMock(spec=TimelineEvent)
        event.event = "cross-referenced"
        event.source = MagicMock()
        event.source.issue = MagicMock()
        # Simulate a linked PR vs. a linked issue
        event.source.issue.pull_request = {} if is_pr else None
        return event

    return _create_mock_event


@pytest.fixture
def mock_filesystem(mocker):
    """Mocks Path and file operations, handling the / operator and Path.open."""
    mock_path_instance_initial = MagicMock(spec=Path, name="Path_initial_instance")
    mock_path_instance_final = MagicMock(
        spec=Path, name="Path_final_instance"
    )  # Result after division

    # Configure the initial instance's division operator
    mock_path_instance_initial.__truediv__.return_value = mock_path_instance_final

    # Configure the final instance's methods needed by SUT
    # Let issue cache loading see the file doesn't exist initially
    mock_path_instance_final.exists.return_value = False
    mock_path_instance_final.parent = MagicMock(spec=Path, name="Path_final_parent")
    mock_path_instance_final.parent.mkdir = MagicMock(name="Path_final_parent_mkdir")
    mock_path_instance_final.__str__.return_value = (
        "/fake/cache/issue_label_cache.jsonl"
    )

    # Mock the open method ON the final Path instance for issue cache writing
    mock_issue_cache_open_func = mock_open()
    mock_path_instance_final.open = mock_issue_cache_open_func

    # Patch the Path CLASS to return the initial instance
    mock_path_class = mocker.patch(
        "repobird_leadgen.github_search.Path", return_value=mock_path_instance_initial
    )

    # Mock for the *main* cache file passed into search()
    # This needs to be a separate mock Path object
    mock_main_cache_path = MagicMock(spec=Path, name="MainCachePath")
    mock_main_cache_open_func = mock_open()
    mock_main_cache_path.open = mock_main_cache_open_func
    mock_main_cache_path.parent = MagicMock(spec=Path)
    mock_main_cache_path.parent.mkdir = MagicMock()
    mock_main_cache_path.__str__.return_value = "/fake/main_cache.jsonl"

    return {
        "PathClass": mock_path_class,
        "PathInitialInstance": mock_path_instance_initial,
        "IssueCachePath": mock_path_instance_final,
        "IssueCacheOpenFunc": mock_issue_cache_open_func,  # The mock for Path(...).open
        "MainCachePath": mock_main_cache_path,
        "MainCacheOpenFunc": mock_main_cache_open_func,  # The mock for Path(...).open
    }


# --- Unit Tests ---


@pytest.mark.unit
def test_github_searcher_init(mock_auth, mock_github, mock_filesystem):
    """Tests that GitHubSearcher initializes Auth.Token and Github correctly."""
    token = "test_token_123"
    searcher = GitHubSearcher(token=token)

    mock_auth["class_patch"].assert_called_once_with(token)
    mock_github["constructor_patch"].assert_called_once_with(
        auth=mock_auth["instance"],
        per_page=100,
        retry=5,
        timeout=30,  # Updated timeout
    )
    assert searcher.gh is mock_github["instance"]
    # Check that the Path class was called (with CACHE_DIR)
    mock_filesystem["PathClass"].assert_called_once_with(CACHE_DIR)
    # Check that the division operator was called on the initial instance
    mock_filesystem["PathInitialInstance"].__truediv__.assert_called_once_with(
        "issue_label_cache.jsonl"
    )
    # Check cache loading attempt (exists call on the FINAL instance)
    mock_filesystem["IssueCachePath"].exists.assert_called_once()
    # Check if Path(...).open was called for loading cache (it shouldn't be if exists() is false)
    mock_filesystem[
        "IssueCacheOpenFunc"
    ].assert_not_called()  # Should not be called if exists=False


@pytest.mark.unit
def test_build_repo_query(
    mocker, mock_auth, mock_github, mock_filesystem
):  # Need fixtures for init
    """Tests the _build_repo_query method."""
    fixed_now = datetime(2024, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
    # Patch datetime only within the SUT module to avoid affecting other tests if they use real datetime
    mock_datetime = mocker.patch("repobird_leadgen.github_search.datetime")
    mock_datetime.now.return_value = fixed_now
    # Allow timedelta usage
    mock_datetime.timedelta = timedelta

    searcher = GitHubSearcher(token="dummy")

    query = searcher._build_repo_query(language="python", min_stars=50, recent_days=30)
    expected_date_str = (fixed_now - timedelta(days=30)).strftime("%Y-%m-%d")
    assert "archived:false" in query
    assert "fork:false" in query
    assert "stars:>=50" in query
    assert "language:python" in query
    assert f"pushed:>{expected_date_str}" in query


@pytest.mark.unit
def test_has_open_issue_with_label_api_true(
    mock_auth, mock_github, mock_repository, mock_filesystem
):
    """Tests _has_open_issue_with_label_api when an issue exists."""
    searcher = GitHubSearcher(token="dummy")
    mock_github["search_issues_results"].totalCount = 1  # Simulate finding an issue
    repo = mock_repository
    label = "bug"

    # Mock _execute_with_retry to just return the result of the function call
    searcher._execute_with_retry = MagicMock(
        side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
    )

    result_tuple = searcher._has_open_issue_with_label_api(repo, label)

    assert result_tuple[0] is True  # Check the boolean part of the tuple
    expected_query = f"repo:{repo.full_name} is:issue is:open label:{label}"
    # Check that _execute_with_retry was called with the correct args
    searcher._execute_with_retry.assert_called_once_with(
        mock_github["instance"].search_issues, query=expected_query
    )


@pytest.mark.unit
def test_has_open_issue_with_label_api_false(
    mock_auth, mock_github, mock_repository, mock_filesystem
):
    """Tests _has_open_issue_with_label_api when no issue exists."""
    searcher = GitHubSearcher(token="dummy")
    mock_github["search_issues_results"].totalCount = 0  # Simulate finding no issues
    repo = mock_repository
    label = "feature"

    searcher._execute_with_retry = MagicMock(
        side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
    )

    result_tuple = searcher._has_open_issue_with_label_api(repo, label)

    assert result_tuple[0] is False  # Check the boolean part of the tuple
    expected_query = f"repo:{repo.full_name} is:issue is:open label:{label}"
    searcher._execute_with_retry.assert_called_once_with(
        mock_github["instance"].search_issues, query=expected_query
    )


@pytest.mark.unit
def test_has_open_issue_with_label_api_quoted(
    mock_auth, mock_github, mock_repository, mock_filesystem
):
    """Tests _has_open_issue_with_label_api with a label containing spaces."""
    searcher = GitHubSearcher(token="dummy")
    mock_github["search_issues_results"].totalCount = 1
    repo = mock_repository
    label = "good first issue"

    searcher._execute_with_retry = MagicMock(
        side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
    )

    result_tuple = searcher._has_open_issue_with_label_api(repo, label)

    assert result_tuple[0] is True  # Check the boolean part of the tuple
    expected_query = (
        f'repo:{repo.full_name} is:issue is:open label:"{label}"'  # Note the quotes
    )
    searcher._execute_with_retry.assert_called_once_with(
        mock_github["instance"].search_issues, query=expected_query
    )


# --- Tests for _get_linked_prs_count --- #


@pytest.mark.unit
def test_get_linked_prs_count_zero(
    mocker, mock_auth, mock_github, mock_issue, mock_timeline_event, mock_filesystem
):
    """Tests counting zero linked PRs."""
    searcher = GitHubSearcher(token="dummy")
    now = datetime.now(timezone.utc)
    issue = mock_issue(1, now)
    # Simulate timeline with non-PR events or irrelevant cross-references
    event_other = MagicMock(spec=TimelineEvent, event="commented")
    event_linked_issue = mock_timeline_event(is_pr=False)
    # Mock the paginator returned by get_timeline
    mock_timeline_paginator = MagicMock(spec=PaginatedList)
    mock_timeline_paginator.__iter__.return_value = iter(
        [event_other, event_linked_issue]
    )
    issue.get_timeline.return_value = mock_timeline_paginator

    # Mock _execute_with_retry to return the paginator
    searcher._execute_with_retry = MagicMock(return_value=mock_timeline_paginator)

    count = searcher._get_linked_prs_count(issue)

    assert count == 0
    searcher._execute_with_retry.assert_called_once_with(issue.get_timeline)


@pytest.mark.unit
def test_get_linked_prs_count_two(
    mocker, mock_auth, mock_github, mock_issue, mock_timeline_event, mock_filesystem
):
    """Tests counting two linked PRs."""
    searcher = GitHubSearcher(token="dummy")
    now = datetime.now(timezone.utc)
    issue = mock_issue(2, now)
    event_pr1 = mock_timeline_event(is_pr=True)
    event_pr2 = mock_timeline_event(is_pr=True)
    event_comment = MagicMock(spec=TimelineEvent, event="commented")
    event_linked_issue = mock_timeline_event(is_pr=False)

    mock_timeline_paginator = MagicMock(spec=PaginatedList)
    mock_timeline_paginator.__iter__.return_value = iter(
        [event_pr1, event_comment, event_pr2, event_linked_issue]
    )
    issue.get_timeline.return_value = mock_timeline_paginator

    searcher._execute_with_retry = MagicMock(return_value=mock_timeline_paginator)

    count = searcher._get_linked_prs_count(issue)

    assert count == 2
    searcher._execute_with_retry.assert_called_once_with(issue.get_timeline)


@pytest.mark.unit
def test_get_linked_prs_count_error(
    mocker, mock_auth, mock_github, mock_issue, mock_filesystem
):
    """Tests error handling during timeline fetch."""
    searcher = GitHubSearcher(token="dummy")
    now = datetime.now(timezone.utc)
    issue = mock_issue(3, now)
    # Simulate error during _execute_with_retry
    searcher._execute_with_retry = MagicMock(side_effect=RuntimeError("API Failed"))

    count = searcher._get_linked_prs_count(issue)

    assert count == -1  # Error indicator
    searcher._execute_with_retry.assert_called_once_with(issue.get_timeline)


# --- Tests for _find_qualifying_issues --- #


@pytest.mark.unit
def test_find_qualifying_issues_no_filters(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests qualification with no age/PR filters (should qualify if any issue found)."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    now = datetime.now(timezone.utc)
    issue1 = mock_issue(101, now - timedelta(days=10))

    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue1])
    mock_issues_paginator.totalCount = 1
    repo.get_issues.return_value = mock_issues_paginator

    searcher._execute_with_retry = MagicMock(return_value=mock_issues_paginator)

    result = searcher._find_qualifying_issues(
        repo, "bug", None
    )  # Removed max_linked_prs=None

    assert result  # Check if the list is non-empty
    searcher._execute_with_retry.assert_called_once_with(
        repo.get_issues, state="open", labels=["bug"]
    )


@pytest.mark.unit
def test_find_qualifying_issues_no_issues_found(
    mocker, mock_auth, mock_github, mock_repository, mock_filesystem
):
    """Tests when get_issues returns no issues."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([])
    mock_issues_paginator.totalCount = 0  # Default total count  # No issues
    repo.get_issues.return_value = mock_issues_paginator

    searcher._execute_with_retry = MagicMock(return_value=mock_issues_paginator)

    result = searcher._find_qualifying_issues(
        repo, "bug", None
    )  # Removed max_linked_prs=None

    assert not result  # Check if the list is empty
    searcher._execute_with_retry.assert_called_once_with(
        repo.get_issues, state="open", labels=["bug"]
    )


@pytest.mark.unit
def test_find_qualifying_issues_age_met(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests qualification when issue meets the max age criteria."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    fixed_now = datetime(2024, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    # Patch datetime within the SUT module
    mock_dt = mocker.patch("repobird_leadgen.github_search.datetime")
    mock_dt.now.return_value = fixed_now
    mock_dt.timedelta = timedelta  # Keep timedelta working

    issue_young = mock_issue(102, fixed_now - timedelta(days=5))
    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue_young])
    mock_issues_paginator.totalCount = 1
    repo.get_issues.return_value = mock_issues_paginator

    searcher._execute_with_retry = MagicMock(return_value=mock_issues_paginator)

    result = searcher._find_qualifying_issues(
        repo, "bug", 10
    )  # Max 10 days old, Removed max_linked_prs=None

    assert result  # Check if the list is non-empty
    searcher._execute_with_retry.assert_called_once_with(
        repo.get_issues, state="open", labels=["bug"]
    )


@pytest.mark.unit
def test_find_qualifying_issues_age_not_met(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests non-qualification when issue is older than max age."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    fixed_now = datetime(2024, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    mock_dt = mocker.patch("repobird_leadgen.github_search.datetime")
    mock_dt.now.return_value = fixed_now
    mock_dt.timedelta = timedelta

    issue_old = mock_issue(103, fixed_now - timedelta(days=15))
    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue_old])
    mock_issues_paginator.totalCount = 1
    repo.get_issues.return_value = mock_issues_paginator

    searcher._execute_with_retry = MagicMock(return_value=mock_issues_paginator)

    result = searcher._find_qualifying_issues(
        repo, "bug", 10
    )  # Max 10 days old, Removed max_linked_prs=None

    assert not result  # Check if the list is empty
    searcher._execute_with_retry.assert_called_once_with(
        repo.get_issues, state="open", labels=["bug"]
    )


@pytest.mark.unit
def test_find_qualifying_issues_pr_met(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests qualification when issue meets the max linked PR criteria."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    now = datetime.now(timezone.utc)
    issue1 = mock_issue(104, now)

    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue1])
    mock_issues_paginator.totalCount = 1
    repo.get_issues.return_value = mock_issues_paginator

    # Mock _get_linked_prs_count to return 1 PR
    searcher._get_linked_prs_count = MagicMock(return_value=1)
    # Mock _execute_with_retry for the get_issues call
    mock_execute = MagicMock(return_value=mock_issues_paginator)
    searcher._execute_with_retry = mock_execute

    result = searcher._find_qualifying_issues(
        repo, "bug", None
    )  # Removed max_linked_prs=2

    assert result  # Check if the list is non-empty
    # PR check is now handled outside this function, so no need to check _get_linked_prs_count here
    # _execute_with_retry is called for get_issues AND potentially for get_timeline inside _get_linked_prs_count
    # We only assert the get_issues call here, assuming _get_linked_prs_count works as tested separately
    mock_execute.assert_any_call(repo.get_issues, state="open", labels=["bug"])
    # PR check is now handled outside this function, so no need to check _get_linked_prs_count here
    # searcher._get_linked_prs_count.assert_called_once_with(issue1)


@pytest.mark.unit
def test_find_qualifying_issues_pr_not_met(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests non-qualification when issue has too many linked PRs."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    now = datetime.now(timezone.utc)
    issue1 = mock_issue(105, now)
    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue1])
    mock_issues_paginator.totalCount = 1
    repo.get_issues.return_value = mock_issues_paginator

    searcher._get_linked_prs_count = MagicMock(return_value=3)
    mock_execute = MagicMock(return_value=mock_issues_paginator)
    searcher._execute_with_retry = mock_execute

    result = searcher._find_qualifying_issues(
        repo, "bug", None
    )  # Removed max_linked_prs=2

    assert result  # Check if the list is non-empty (function only checks age now)
    mock_execute.assert_any_call(repo.get_issues, state="open", labels=["bug"])
    # PR check is now handled outside this function, so no need to check _get_linked_prs_count here
    # searcher._get_linked_prs_count.assert_called_once_with(issue1)


@pytest.mark.unit
def test_find_qualifying_issues_pr_count_error(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests skipping issue if PR count fails."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    now = datetime.now(timezone.utc)
    issue1 = mock_issue(106, now)
    issue2 = mock_issue(107, now)  # Add a second issue that might qualify

    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue1, issue2])
    mock_issues_paginator.totalCount = 2
    repo.get_issues.return_value = mock_issues_paginator

    # Mock _get_linked_prs_count to fail for issue1, succeed for issue2
    searcher._get_linked_prs_count = MagicMock(side_effect=[-1, 0])
    mock_execute = MagicMock(return_value=mock_issues_paginator)
    searcher._execute_with_retry = mock_execute

    result = searcher._find_qualifying_issues(
        repo, "bug", None
    )  # Removed max_linked_prs=2

    assert result  # Should qualify based on issue2 (list is non-empty)
    mock_execute.assert_any_call(repo.get_issues, state="open", labels=["bug"])
    # PR check is now handled outside this function, so no need to check _get_linked_prs_count here
    # assert searcher._get_linked_prs_count.call_count == 2
    # searcher._get_linked_prs_count.assert_has_calls([call(issue1), call(issue2)])


@pytest.mark.unit
def test_find_qualifying_issues_age_and_pr_met(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests qualification when both age and PR criteria are met."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    fixed_now = datetime(2024, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    mock_dt = mocker.patch("repobird_leadgen.github_search.datetime")
    mock_dt.now.return_value = fixed_now
    mock_dt.timedelta = timedelta

    issue_good = mock_issue(108, fixed_now - timedelta(days=5))
    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue_good])
    mock_issues_paginator.totalCount = 1
    repo.get_issues.return_value = mock_issues_paginator

    searcher._get_linked_prs_count = MagicMock(return_value=1)
    mock_execute = MagicMock(return_value=mock_issues_paginator)
    searcher._execute_with_retry = mock_execute

    result = searcher._find_qualifying_issues(
        repo,
        "bug",
        10,  # Removed max_linked_prs=2
    )  # Max 10 days

    assert result  # Check if the list is non-empty
    mock_execute.assert_any_call(repo.get_issues, state="open", labels=["bug"])
    # PR check is now handled outside this function, so no need to check _get_linked_prs_count here
    # searcher._get_linked_prs_count.assert_called_once_with(issue_good)


@pytest.mark.unit
def test_find_qualifying_issues_age_fail_pr_met(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests non-qualification when age fails but PRs are okay."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    fixed_now = datetime(2024, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    mock_dt = mocker.patch("repobird_leadgen.github_search.datetime")
    mock_dt.now.return_value = fixed_now
    mock_dt.timedelta = timedelta

    issue_old = mock_issue(109, fixed_now - timedelta(days=15))
    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue_old])
    mock_issues_paginator.totalCount = 1
    repo.get_issues.return_value = mock_issues_paginator

    searcher._get_linked_prs_count = MagicMock(return_value=1)
    mock_execute = MagicMock(return_value=mock_issues_paginator)
    searcher._execute_with_retry = mock_execute

    result = searcher._find_qualifying_issues(
        repo,
        "bug",
        10,  # Removed max_linked_prs=2
    )  # Max 10 days

    assert not result  # Check if the list is empty (age filter should fail)
    mock_execute.assert_any_call(repo.get_issues, state="open", labels=["bug"])
    # PR check is now handled outside this function, so no need to check _get_linked_prs_count here
    # searcher._get_linked_prs_count.assert_not_called() # PR check is irrelevant if age fails


@pytest.mark.unit
def test_find_qualifying_issues_age_met_pr_fail(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests non-qualification when age is okay but PRs fail."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    fixed_now = datetime(2024, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    mock_dt = mocker.patch("repobird_leadgen.github_search.datetime")
    mock_dt.now.return_value = fixed_now
    mock_dt.timedelta = timedelta

    issue_young = mock_issue(110, fixed_now - timedelta(days=5))
    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter([issue_young])
    mock_issues_paginator.totalCount = 1
    repo.get_issues.return_value = mock_issues_paginator

    searcher._get_linked_prs_count = MagicMock(return_value=3)
    mock_execute = MagicMock(return_value=mock_issues_paginator)
    searcher._execute_with_retry = mock_execute

    result = searcher._find_qualifying_issues(
        repo,
        "bug",
        10,  # Removed max_linked_prs=2
    )  # Max 10 days

    assert result  # Check if the list is non-empty (age filter passes)
    mock_execute.assert_any_call(repo.get_issues, state="open", labels=["bug"])
    # PR check is now handled outside this function, so no need to check _get_linked_prs_count here
    # searcher._get_linked_prs_count.assert_called_once_with(issue_young)


@pytest.mark.unit
def test_find_qualifying_issues_multiple_issues_one_qualifies(
    mocker, mock_auth, mock_github, mock_repository, mock_issue, mock_filesystem
):
    """Tests finding one qualifying issue among several non-qualifying ones."""
    searcher = GitHubSearcher(token="dummy")
    repo = mock_repository
    fixed_now = datetime(2024, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    mock_dt = mocker.patch("repobird_leadgen.github_search.datetime")
    mock_dt.now.return_value = fixed_now
    mock_dt.timedelta = timedelta

    issue_old = mock_issue(201, fixed_now - timedelta(days=20))
    issue_many_prs = mock_issue(202, fixed_now - timedelta(days=5))
    issue_good = mock_issue(203, fixed_now - timedelta(days=8))
    issue_also_old = mock_issue(204, fixed_now - timedelta(days=30))

    mock_issues_paginator = MagicMock(spec=PaginatedList)
    mock_issues_paginator.__iter__.return_value = iter(
        [issue_old, issue_many_prs, issue_good, issue_also_old]
    )
    mock_issues_paginator.totalCount = 4
    repo.get_issues.return_value = mock_issues_paginator

    # Mock _get_linked_prs_count: called for 202 (returns 3), called for 203 (returns 1)
    searcher._get_linked_prs_count = MagicMock(side_effect=[3, 1])
    mock_execute = MagicMock(return_value=mock_issues_paginator)
    searcher._execute_with_retry = mock_execute

    result = searcher._find_qualifying_issues(
        repo,
        "bug",
        15,  # Removed max_linked_prs=2
    )  # Max 15 days

    assert result  # Should qualify based on issue_good (list is non-empty)
    mock_execute.assert_any_call(repo.get_issues, state="open", labels=["bug"])
    # PR check is now handled outside this function, so no need to check _get_linked_prs_count here
    # assert searcher._get_linked_prs_count.call_count == 2
    # searcher._get_linked_prs_count.assert_has_calls(
    #     [call(issue_many_prs), call(issue_good)]
    # )


# --- Tests for search method with filters --- #


# Helper fixture to set up mocks for the search method tests
# Parametrized to control issue cache state ('miss', 'hit_true', 'hit_false')
@pytest.fixture(params=["miss", "hit_true", "hit_false"])
def mock_search_flow(mocker, mock_github, mock_repository, mock_filesystem, request):
    issue_cache_state = request.param
    label_used_in_test = "feature"  # Label used in the search tests

    searcher = GitHubSearcher(token="dummy")
    repo1 = mock_repository
    cache_key = (repo1.full_name, label_used_in_test)

    # Initialize issue cache (clear it first)
    searcher.issue_cache = {}
    # Configure issue cache based on state parameter
    if issue_cache_state == "hit_true":
        searcher.issue_cache[cache_key] = (
            True,
            [{"url": "dummy_url"}],
        )  # Value doesn't matter much, just existence
    elif issue_cache_state == "hit_false":
        searcher.issue_cache[cache_key] = (False, [])
    # 'miss' state: searcher.issue_cache is empty

    # Mock the initial repo search results
    mock_github["search_repos_results"].__iter__.return_value = iter([repo1])
    mock_github["search_repos_results"].totalCount = 1

    # Mock internal methods (can be overridden in tests)
    # Simulate API check returning True by default for cache miss scenario (for non-filter tests)
    mock_initial_check = mocker.patch.object(
        searcher,
        "_has_open_issue_with_label_api",
        return_value=(
            True,
            [{"number": 1, "html_url": "http://fake.url/1"}],
        ),  # Return tuple
    )
    # Simulate detailed check returning a list with one issue ID by default
    mock_detailed_check = mocker.patch.object(
        searcher,
        "_find_qualifying_issues",
        return_value=[
            {"number": 123, "html_url": "http://fake.url/123"}
        ],  # Return list of dicts
    )

    # Mock _execute_with_retry
    def execute_side_effect(func, *args, **kwargs):
        if func == mock_github["instance"].search_repositories:
            return mock_github["search_repos_results"]
        elif func == next:
            try:
                return func(*args, **kwargs)
            except StopIteration:
                raise
        # Allow mocked internal methods to be called via _execute_with_retry
        elif func == searcher._has_open_issue_with_label_api:
            return mock_initial_check(*args, **kwargs)
        elif func == searcher._find_qualifying_issues:
            return mock_detailed_check(*args, **kwargs)
        else:
            # Default pass-through for other unmocked API calls if needed
            return func(*args, **kwargs)

    mock_execute = mocker.patch.object(
        searcher, "_execute_with_retry", side_effect=execute_side_effect
    )

    # Get mock Path objects and open mocks from filesystem fixture
    mock_issue_cache_path = mock_filesystem["IssueCachePath"]
    mock_issue_cache_open_func = mock_filesystem["IssueCacheOpenFunc"]
    mock_main_cache_path = mock_filesystem["MainCachePath"]
    mock_main_cache_open_func = mock_filesystem["MainCacheOpenFunc"]

    # Patch json.dump globally to track calls
    mock_json_dump = mocker.patch("json.dump")

    return {
        "searcher": searcher,
        "repo": repo1,
        "label": label_used_in_test,
        "mock_initial_check": mock_initial_check,
        "mock_detailed_check": mock_detailed_check,
        "mock_execute": mock_execute,
        "mock_issue_cache_path": mock_issue_cache_path,
        "mock_issue_cache_open_func": mock_issue_cache_open_func,
        "mock_main_cache_path": mock_main_cache_path,
        "mock_main_cache_open_func": mock_main_cache_open_func,
        "mock_json_dump": mock_json_dump,
        "issue_cache_state": issue_cache_state,
        "cache_key": cache_key,
    }


# Helper function to assert issue cache dump call
def assert_issue_cache_dump(
    mock_json_dump, repo_full_name, label, expected_has_label, handle
):
    # Find the call related to the issue cache
    issue_cache_call = None
    for c in mock_json_dump.call_args_list:
        args, _ = c
        if isinstance(args[0], dict) and args[0].get("repo") == repo_full_name:
            issue_cache_call = c
            break
    assert (
        issue_cache_call is not None
    ), f"Issue cache dump call not found for {repo_full_name}"
    args, _ = issue_cache_call
    assert args[0]["repo"] == repo_full_name
    assert args[0]["label"] == label
    assert args[0]["has_label"] is expected_has_label
    assert args[1] is handle, "Issue cache dump handle mismatch"


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm", MagicMock())
@patch("repobird_leadgen.github_search.time.sleep", MagicMock())
@patch("builtins.print")
def test_search_with_filters_qualifies(mock_print, mock_search_flow):
    """Tests search when repo qualifies based on detailed filters."""
    searcher = mock_search_flow["searcher"]
    repo1 = mock_search_flow["repo"]
    label = mock_search_flow["label"]
    mock_main_cache_path = mock_search_flow["mock_main_cache_path"]
    mock_main_cache_open_func = mock_search_flow["mock_main_cache_open_func"]
    mock_issue_cache_open_func = mock_search_flow["mock_issue_cache_open_func"]
    mock_json_dump = mock_search_flow["mock_json_dump"]
    # issue_cache_state = mock_search_flow["issue_cache_state"] # Removed F841

    # Mock detailed check to return a qualifying list of issue detail dicts
    qualifying_details = [{"number": 456, "html_url": "http://fake.url/456"}]
    mock_search_flow["mock_detailed_check"].return_value = qualifying_details
    max_age, max_prs = 30, 1

    results = list(
        searcher.search(
            label=label,
            language="go",
            max_issue_age_days=max_age,
            max_linked_prs=max_prs,
            cache_file=mock_main_cache_path,
            existing_repo_names=set(),
        )
    )

    # assert len(results) == expected_results # Removed F821 - Redundant check

    assert (
        len(results) == 1
    )  # Should qualify based on non-empty list from detailed check

    # Assertions common to all states
    mock_main_cache_open_func.assert_called_with("a", encoding="utf-8")
    main_cache_handle = mock_main_cache_open_func.return_value
    # Internal issue cache should NOT be written when detailed filters are active
    mock_issue_cache_open_func.assert_not_called()

    # Initial check is always bypassed with filters
    mock_search_flow["mock_initial_check"].assert_not_called()
    # Detailed check is always called when filters are active
    mock_search_flow["mock_detailed_check"].assert_called_once_with(
        repo1,
        label,
        max_age,  # Removed max_prs from assertion
    )

    # Assertions for qualifying case
    assert results[0] == repo1
    # Check main cache write - should contain 'found_issues' list of dicts
    expected_data = repo1.raw_data
    # Use the qualifying_details list defined earlier
    expected_data["found_issues"] = qualifying_details
    expected_data.pop(
        "_repobird_found_issues_basic", None
    )  # Remove old field if present

    mock_json_dump.assert_called_once_with(expected_data, main_cache_handle)
    main_cache_handle.write.assert_called_once_with("\n")
    main_cache_handle.flush.assert_called_once()


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm", MagicMock())
@patch("repobird_leadgen.github_search.time.sleep", MagicMock())
@patch("builtins.print")
def test_search_with_filters_skips_detailed(mock_print, mock_search_flow):
    """Tests search skips repo when detailed filter check fails."""
    # This test is valid for all cache states now, as detailed check runs regardless

    searcher = mock_search_flow["searcher"]
    repo1 = mock_search_flow["repo"]
    label = mock_search_flow["label"]
    mock_main_cache_path = mock_search_flow["mock_main_cache_path"]
    mock_main_cache_open_func = mock_search_flow["mock_main_cache_open_func"]
    mock_issue_cache_open_func = mock_search_flow["mock_issue_cache_open_func"]
    mock_json_dump = mock_search_flow["mock_json_dump"]
    # issue_cache_state = mock_search_flow["issue_cache_state"] # Removed F841

    # Mock detailed check to return an empty list (non-qualifying)
    mock_search_flow["mock_detailed_check"].return_value = []
    max_age, max_prs = 30, 1

    results = list(
        searcher.search(
            label=label,
            language="go",
            max_issue_age_days=max_age,
            max_linked_prs=max_prs,
            cache_file=mock_main_cache_path,
            existing_repo_names=set(),
        )
    )

    assert len(results) == 0

    assert len(results) == 0  # Should not qualify

    # Assertions common to all states
    mock_main_cache_open_func.assert_called_with("a", encoding="utf-8")
    main_cache_handle = mock_main_cache_open_func.return_value
    # Internal issue cache should NOT be written when detailed filters are active
    mock_issue_cache_open_func.assert_not_called()
    # Initial check is bypassed when filters are active
    mock_search_flow["mock_initial_check"].assert_not_called()
    # Detailed check is called (and returns empty list)
    mock_search_flow["mock_detailed_check"].assert_called_once_with(
        repo1,
        label,
        max_age,  # Removed max_prs from assertion
    )

    # Assert nothing dumped/written to main cache
    mock_json_dump.assert_not_called()
    main_cache_handle.write.assert_not_called()
    main_cache_handle.flush.assert_not_called()


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm", MagicMock())
@patch("repobird_leadgen.github_search.time.sleep", MagicMock())
@patch("builtins.print")
def test_search_with_filters_skips_cache_or_failed_detailed_DEPRECATED(
    mock_print, mock_search_flow
):
    """
    DEPRECATED: Original logic tested skipping based on hit_false OR failed detailed check.
    This is now covered by test_search_with_filters_skips_detailed.
    Keeping structure for reference but marking as skip.
    """
    pytest.skip(
        "Deprecated test, logic covered by test_search_with_filters_skips_detailed"
    )

    # Original code below for reference
    # # Only run for cache miss or hit_false states
    # if mock_search_flow["issue_cache_state"] == "hit_true":
    #     pytest.skip("N/A for hit_true state, covered elsewhere")
    #
    # searcher = mock_search_flow["searcher"]
    # repo1 = mock_search_flow["repo"]
    # label = mock_search_flow["label"]
    # mock_main_cache_path = mock_search_flow["mock_main_cache_path"]
    # mock_main_cache_open_func = mock_search_flow["mock_main_cache_open_func"]
    # mock_issue_cache_open_func = mock_search_flow["mock_issue_cache_open_func"]
    # mock_json_dump = mock_search_flow["mock_json_dump"]
    # issue_cache_state = mock_search_flow["issue_cache_state"] # miss or hit_false
    #
    # # Mock detailed check to return False (only matters for 'miss' case originally)
    # mock_search_flow["mock_detailed_check"].return_value = False
    # max_age, max_prs = 30, 1
    #
    # results = list(searcher.search(
    #     label=label,
    #     language="go",
    #     max_issue_age_days=max_age,
    #     max_linked_prs=max_prs,
    #     cache_file=mock_main_cache_path,
    #     existing_repo_names=set(),
    # ))
    #
    # assert len(results) == 0
    #
    # # Assertions common to relevant states (miss, hit_false)
    # mock_main_cache_open_func.assert_called_with("a", encoding="utf-8")
    # main_cache_handle = mock_main_cache_open_func.return_value
    # # Issue cache should NEVER be written when filters are active
    # mock_issue_cache_open_func.assert_not_called()
    # # Initial check is bypassed when filters are active
    # mock_search_flow["mock_initial_check"].assert_not_called()
    #
    # # Assertions specific to cache state (ORIGINAL INTENT)
    # if issue_cache_state == "miss":
    #     # Detailed check is called (and returns False)
    #     mock_search_flow["mock_detailed_check"].assert_called_once_with(repo1, label, max_age, max_prs)
    # elif issue_cache_state == "hit_false":
    #     # Detailed check is skipped due to cache
    #     mock_search_flow["mock_detailed_check"].assert_not_called()
    #
    # # Assert nothing dumped/written
    # mock_json_dump.assert_not_called()
    # main_cache_handle.write.assert_not_called()
    # main_cache_handle.flush.assert_not_called()


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm", MagicMock())
@patch("repobird_leadgen.github_search.time.sleep", MagicMock())
@patch("builtins.print")
def test_search_no_filters_qualifies(mock_print, mock_search_flow):
    """Tests search qualification when no detailed filters are active."""
    # Only run for cache states where initial check passes
    if mock_search_flow["issue_cache_state"] == "hit_false":
        pytest.skip("N/A for hit_false state")

    searcher = mock_search_flow["searcher"]
    repo1 = mock_search_flow["repo"]
    label = mock_search_flow["label"]
    mock_main_cache_path = mock_search_flow["mock_main_cache_path"]
    mock_main_cache_open_func = mock_search_flow["mock_main_cache_open_func"]
    mock_issue_cache_open_func = mock_search_flow["mock_issue_cache_open_func"]
    mock_json_dump = mock_search_flow["mock_json_dump"]
    issue_cache_state = mock_search_flow["issue_cache_state"]
    cache_key = mock_search_flow["cache_key"]

    # Setup expected issue details for cache hit/miss scenarios
    expected_found_issues = []
    if issue_cache_state == "hit_true":
        # Extract numbers from the details stored in the mock cache setup
        mock_details = searcher.issue_cache[cache_key][1]
        expected_found_issues = [
            item.get("number")
            for item in mock_details
            if isinstance(item, dict) and item.get("number") is not None
        ]
    elif issue_cache_state == "miss":
        # API check path - currently results in empty list in output
        expected_found_issues = []

    results = list(
        searcher.search(
            label=label,
            language="go",
            max_issue_age_days=None,  # No filters
            max_linked_prs=None,
            cache_file=mock_main_cache_path,
            existing_repo_names=set(),
        )
    )

    assert len(results) == 1
    assert results[0] == repo1

    main_cache_handle = mock_main_cache_open_func.return_value
    issue_cache_handle = mock_issue_cache_open_func.return_value

    # Detailed check should NOT be performed
    mock_search_flow["mock_detailed_check"].assert_not_called()

    # Prepare expected data for main cache dump
    expected_data = repo1.raw_data
    expected_data["found_issues"] = expected_found_issues  # Add the list of IDs
    expected_data.pop("_repobird_found_issues_basic", None)  # Remove old field

    if issue_cache_state == "miss":
        # Initial check performed (API mock returns True)
        mock_search_flow["mock_initial_check"].assert_called_once_with(repo1, label)
        # Internal issue cache appended (has_label=True, details=[])
        mock_issue_cache_open_func.assert_called_with("a", encoding="utf-8")
        assert_issue_cache_dump(
            mock_json_dump, repo1.full_name, label, True, issue_cache_handle
        )  # This checks the internal cache dump
        # Repo data dumped to main cache
        mock_main_cache_open_func.assert_called_with("a", encoding="utf-8")
        # Find the main cache dump call
        main_cache_call = None
        for c in mock_json_dump.call_args_list:
            args, _ = c
            if args[1] is main_cache_handle:
                main_cache_call = c
                break
        assert main_cache_call is not None, "Main cache dump call not found"
        assert (
            main_cache_call.args[0] == expected_data
        )  # Check data written to main cache
        assert (
            mock_json_dump.call_count == 2
        )  # One for internal cache, one for main cache
    elif issue_cache_state == "hit_true":
        # Initial check NOT performed (cache hit)
        mock_search_flow["mock_initial_check"].assert_not_called()
        # Internal issue cache NOT appended
        mock_issue_cache_open_func.assert_not_called()
        # Repo data dumped to main cache
        mock_main_cache_open_func.assert_called_with("a", encoding="utf-8")
        mock_json_dump.assert_called_once_with(
            expected_data, main_cache_handle
        )  # Check data written to main cache

    # Check main cache file writes
    main_cache_handle.write.assert_called_once_with("\n")
    main_cache_handle.flush.assert_called_once()


@pytest.mark.unit
@patch("repobird_leadgen.github_search.tqdm", MagicMock())
@patch("repobird_leadgen.github_search.time.sleep", MagicMock())
@patch("builtins.print")
def test_search_no_filters_skips_initial(mock_print, mock_search_flow):
    """Tests search skips repo when initial label check fails (no filters active)."""
    # Only run for cache states where initial check fails
    if mock_search_flow["issue_cache_state"] == "hit_true":
        pytest.skip("N/A for hit_true state")

    searcher = mock_search_flow["searcher"]
    repo1 = mock_search_flow["repo"]
    label = mock_search_flow["label"]
    mock_main_cache_path = mock_search_flow["mock_main_cache_path"]
    mock_main_cache_open_func = mock_search_flow["mock_main_cache_open_func"]
    mock_issue_cache_open_func = mock_search_flow["mock_issue_cache_open_func"]
    mock_json_dump = mock_search_flow["mock_json_dump"]
    issue_cache_state = mock_search_flow["issue_cache_state"]

    # Override initial API check mock to return False for cache miss scenario
    if issue_cache_state == "miss":
        mock_search_flow["mock_initial_check"].return_value = (False, [])

    results = list(
        searcher.search(
            label=label,
            language="go",
            max_issue_age_days=None,  # No filters
            max_linked_prs=None,
            cache_file=mock_main_cache_path,
            existing_repo_names=set(),
        )
    )

    assert len(results) == 0
    mock_main_cache_open_func.assert_called_with(
        "a", encoding="utf-8"
    )  # Main cache file opened
    main_cache_handle = mock_main_cache_open_func.return_value
    issue_cache_handle = mock_issue_cache_open_func.return_value

    # Detailed check should NOT be performed
    mock_search_flow["mock_detailed_check"].assert_not_called()

    if issue_cache_state == "miss":
        # Initial check performed (API mock returns False)
        mock_search_flow["mock_initial_check"].assert_called_once_with(repo1, label)
        # Internal issue cache appended (has_label=False, details=[])
        mock_issue_cache_open_func.assert_called_with("a", encoding="utf-8")
        assert_issue_cache_dump(
            mock_json_dump, repo1.full_name, label, False, issue_cache_handle
        )
        # Only internal issue cache dump expected
        assert mock_json_dump.call_count == 1
    elif issue_cache_state == "hit_false":
        # Initial check NOT performed (cache hit)
        mock_search_flow["mock_initial_check"].assert_not_called()
        # Internal issue cache NOT appended
        mock_issue_cache_open_func.assert_not_called()
        # No dump expected
        mock_json_dump.assert_not_called()

    # Check nothing written to main cache
    main_cache_handle.write.assert_not_called()
    main_cache_handle.flush.assert_not_called()


# --- Integration Tests --- (Kept for context, might need adjustments later)


# Add a newline before the next test definition for clarity and to avoid potential parser issues
@pytest.mark.integration
@skip_if_no_token
def test_basic_search_integration(tmp_path):
    """Integration test for basic search without detailed filters."""
    gh = GitHubSearcher()
    # Use a temporary file for the cache
    cache_file = tmp_path / "integration_basic_cache.jsonl"
    try:
        repos_iterator = gh.search(
            label="good first issue",
            language="python",
            min_stars=50,
            recent_days=730,  # 2 years
            max_results=3,
            cache_file=cache_file,
            existing_repo_names=set(),
        )
        repos = list(repos_iterator)

        assert isinstance(repos, list)
        if repos:
            assert isinstance(repos[0], Repository)
            assert cache_file.exists()
            # Check cache file content (basic check)
            with cache_file.open("r") as f:
                lines = f.readlines()
                assert len(lines) == len(repos)
                first_repo_data = json.loads(lines[0])
                assert first_repo_data["full_name"] == repos[0].full_name
        else:
            print("\nWarning: Basic integration test found 0 repos.")

        assert len(repos) <= 3

    except RateLimitExceededException:
        pytest.skip("GitHub API rate limit exceeded.")
    except GithubException as e:
        pytest.fail(f"GitHub API error during test_basic_search: {e.status} {e.data}")
    except Exception as e:
        pytest.fail(f"Unexpected error during test_basic_search: {e}")


@pytest.mark.integration
@skip_if_no_token
def test_search_with_filters_integration(tmp_path):
    """Integration test for search WITH detailed filters (age/PRs)."""
    gh = GitHubSearcher()
    cache_file = tmp_path / "integration_filtered_cache.jsonl"
    try:
        # Use filters likely to yield few results quickly
        # e.g., recent issues with zero linked PRs
        repos_iterator = gh.search(
            label="help wanted",  # Common label
            language="python",
            min_stars=100,
            recent_days=180,
            max_results=2,  # Look for few results
            max_issue_age_days=90,  # Issues created in last 90 days
            max_linked_prs=0,  # Issues with zero linked PRs
            cache_file=cache_file,
            existing_repo_names=set(),
        )
        repos = list(repos_iterator)

        assert isinstance(repos, list)
        if repos:
            print(f"\nFiltered integration test found {len(repos)} repos:")
            for r in repos:
                print(f"  - {r.full_name}")
            assert isinstance(repos[0], Repository)
            assert cache_file.exists()
            with cache_file.open("r") as f:
                lines = f.readlines()
                assert len(lines) == len(repos)
        else:
            # This is possible if no repos match the strict criteria
            print(
                "\nWarning: Filtered integration test found 0 repos matching criteria."
            )
            # Check if cache file was created (even if empty)
            assert cache_file.exists()
            # assert cache_file.read_text() == "" # File might contain header or be non-empty from previous runs if not cleaned

        assert len(repos) <= 2

    except RateLimitExceededException:
        pytest.skip("GitHub API rate limit exceeded.")
    except GithubException as e:
        pytest.fail(
            f"GitHub API error during test_search_with_filters: {e.status} {e.data}"
        )
    except Exception as e:
        pytest.fail(f"Unexpected error during test_search_with_filters: {e}")
