import os
import pytest
from github import GithubException
from repobird_leadgen.github_search import GitHubSearcher
from repobird_leadgen.config import GITHUB_TOKEN # Import token for check

# Basic check to skip tests if no token is available
# You might want a more sophisticated way to handle this in a real CI environment
REASON_NO_TOKEN = "Skipping integration test: GITHUB_TOKEN not set in environment."
pytestmark = pytest.mark.skipif(not GITHUB_TOKEN, reason=REASON_NO_TOKEN)

# Mark tests that require network access and a valid GitHub Token
@pytest.mark.integration
def test_basic_search_integration():
    """
    Tests a basic search query against the live GitHub API.
    Requires a valid GITHUB_TOKEN.
    """
    gh = GitHubSearcher() # Uses token from environment/config
    try:
        # Perform a minimal search to limit API usage
        repos = gh.search(label="good first issue", language="python", max_results=1, min_stars=1, recent_days=365 * 5) # Broaden criteria slightly for test
        # We expect at least one result for such a broad query, but it's not guaranteed
        # Assert that the call runs without raising exceptions and returns a list
        assert isinstance(repos, list)
        # Optionally, assert len > 0 if you are reasonably sure the query will return results
        # assert len(repos) >= 0 # Check it returns a list, length can be 0

    except GithubException as e:
        # Fail the test if a GitHub API error occurs (like auth failure)
        pytest.fail(f"GitHub API error during test_basic_search: {e.status} {e.data}")
    except Exception as e:
        # Fail on other unexpected errors
        pytest.fail(f"Unexpected error during test_basic_search: {e}")

# Add another test for the helper function if desired
@pytest.mark.integration
def test_good_first_issue_repos_helper_integration():
    """Tests the good_first_issue_repos helper method."""
    gh = GitHubSearcher()
    try:
        repos = gh.good_first_issue_repos(language="python", max_results=1, min_stars=1, recent_days=365*5)
        assert isinstance(repos, list)
    except GithubException as e:
        pytest.fail(f"GitHub API error during test_good_first_issue_repos_helper: {e.status} {e.data}")
    except Exception as e:
        pytest.fail(f"Unexpected error during test_good_first_issue_repos_helper: {e}")

# Example of a unit test (doesn't hit the API)
@pytest.mark.unit
def test_build_query_formatting():
    """Tests the internal _build_query method formatting."""
    gh = GitHubSearcher(token="dummy_token") # Doesn't need a real token
    query = gh._build_query(label="help wanted", language="javascript", min_stars=100, recent_days=90)

    assert 'label:"help wanted"' in query # Check label quoting
    assert 'language:javascript' in query
    assert 'stars:>=100' in query
    # Check date formatting (more complex to test precisely without mocking datetime)
    assert 'pushed:>' in query
    assert 'archived:false' in query
    assert 'fork:false' in query

