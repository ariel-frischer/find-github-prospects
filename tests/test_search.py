import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call, ANY
from datetime import datetime, timedelta, timezone
import github
from github import Github, Auth, RateLimitExceededException, UnknownObjectException, GithubException
from github.Repository import Repository
from github.ContentFile import ContentFile
from github.PaginatedList import PaginatedList
# Import the class from the module under test for patching if needed
from repobird_leadgen import github_search
from repobird_leadgen.github_search import GitHubSearcher
from repobird_leadgen.config import GITHUB_TOKEN # Keep for integration tests
from repobird_leadgen.models import RepoLead, ContactInfo # Import needed models

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
    mock_auth_token_class = mocker.patch('repobird_leadgen.github_search.Auth.Token', return_value=mock_token_instance)
    return {
        'class_patch': mock_auth_token_class, # The mock object returned by patch
        'instance': mock_token_instance     # The mock instance returned by the class patch
    }

@pytest.fixture
def mock_github(mocker):
    """Mocks the Github constructor in the SUT's namespace and provides the instance."""
    mock_instance = MagicMock(spec=Github)
    mock_search_results = MagicMock(spec=PaginatedList)
    # Make the mock iterable
    mock_search_results.__iter__.return_value = iter([]) # Default empty
    mock_instance.search_repositories.return_value = mock_search_results

    # Mock the RateLimit object structure expected by the SUT
    mock_rate_limit_info = MagicMock()
    mock_rate_limit_info.search = MagicMock()
    # Set a realistic reset time slightly in the future
    mock_rate_limit_info.search.reset = datetime.now(timezone.utc) + timedelta(seconds=10)
    mock_instance.get_rate_limit.return_value = mock_rate_limit_info

    # Mock the Github constructor *within the SUT's namespace* ONLY
    mock_constructor_patch = mocker.patch('repobird_leadgen.github_search.Github', return_value=mock_instance)

    return {
        'constructor_patch': mock_constructor_patch, # The mock object returned by patch
        'instance': mock_instance,                 # The mock instance returned by the constructor patch
        'search_results': mock_search_results,       # The mock PaginatedList returned by search_repositories
        'rate_limit_info': mock_rate_limit_info      # The mock rate limit structure
    }

@pytest.fixture
def mock_repo_data():
    """Provides data for a mock Repository object."""
    now = datetime.now(timezone.utc)
    return {
        'full_name': 'test_owner/test_repo',
        'html_url': 'https://github.com/test_owner/test_repo',
        'description': 'A test repository',
        'stargazers_count': 100,
        'forks_count': 20,
        'topics': ['python', 'test'],
        'pushed_at': now,
        'owner': MagicMock(login='test_owner', type='User'),
        'readme_content': b'This is the README content.',
    }

@pytest.fixture
def mock_repository(mocker, mock_repo_data):
    """Creates a mock Repository object."""
    mock_repo = MagicMock(spec=Repository)
    for key, value in mock_repo_data.items():
        if key != 'readme_content':
             setattr(mock_repo, key, value)

    mock_readme_file = MagicMock(spec=ContentFile)
    mock_readme_file.decoded_content = mock_repo_data['readme_content']
    mock_repo.get_readme.return_value = mock_readme_file

    return mock_repo

# --- Unit Tests ---

@pytest.mark.unit
def test_github_searcher_init(mock_auth, mock_github):
    """Tests that GitHubSearcher initializes Auth.Token and Github correctly."""
    token = "test_token_123"
    searcher = GitHubSearcher(token=token) # This call uses the patched constructors

    # Assert the Auth.Token mock (the patched class itself) was called
    mock_auth['class_patch'].assert_called_once_with(token)

    # Assert the Github mock (the patched class itself) was called
    mock_github['constructor_patch'].assert_called_once_with(
        auth=mock_auth['instance'],
        per_page=100, retry=5, timeout=15
    )
    # Assert the instance on the searcher is the one returned by the Github patch
    assert searcher.gh is mock_github['instance']


@pytest.mark.unit
def test_build_query(mocker, mock_auth, mock_github): # Need fixtures for init
    """Tests the _build_query method with various parameters."""
    fixed_now = datetime(2024, 1, 31, 12, 0, 0, tzinfo=timezone.utc)
    # Patch datetime in the SUT's namespace
    mock_datetime = mocker.patch('repobird_leadgen.github_search.datetime')
    mock_datetime.now.return_value = fixed_now
    # Allow datetime constructor usage if needed by SUT or tests
    mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw) if args else fixed_now

    searcher = GitHubSearcher(token="dummy") # Instantiation uses mocks

    query1 = searcher._build_query(label="good first issue", language="python", min_stars=50, recent_days=30)
    expected_date_30 = (fixed_now - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    assert 'label:"good first issue"' in query1
    assert f'pushed:>{expected_date_30}' in query1

    query2 = searcher._build_query(label="help wanted", language="javascript", min_stars=10, recent_days=90)
    expected_date_90 = (fixed_now - timedelta(days=90)).strftime('%Y-%m-%dT%H:%M:%SZ')
    assert 'label:"help wanted"' in query2
    assert f'pushed:>{expected_date_90}' in query2

    query3 = searcher._build_query(label="bug", language="go", min_stars=0, recent_days=1)
    expected_date_1 = (fixed_now - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    assert 'label:"bug"' in query3
    assert f'pushed:>{expected_date_1}' in query3


@pytest.mark.unit
@patch('repobird_leadgen.github_search.tqdm') # Patch tqdm directly
@patch('repobird_leadgen.github_search.time.sleep')
def test_search_success(mock_sleep, mock_tqdm_class, mock_auth, mock_github, mock_repository, mock_repo_data):
    # Configure the mock tqdm class to handle context management
    mock_pbar = MagicMock()
    mock_pbar.update = MagicMock()
    mock_pbar.close = MagicMock()
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_context.__exit__.return_value = None
    mock_tqdm_class.return_value = mock_tqdm_context # Calling tqdm() returns the context manager

    searcher = GitHubSearcher(token="dummy")
    # searcher.gh is already the correct mock instance from the mock_github fixture

    mock_repos_list = [mock_repository]
    # Ensure the search_results mock is iterable
    mock_github['search_results'].__iter__.return_value = iter(mock_repos_list)
    mock_github['search_results'].totalCount = 1

    label, lang, stars, days, max_res = "good first issue", "python", 50, 30, 10
    results = searcher.search(label=label, language=lang, min_stars=stars, recent_days=days, max_results=max_res)

    expected_query = searcher._build_query(label=label, language=lang, min_stars=stars, recent_days=days)
    mock_github['instance'].search_repositories.assert_called_once_with(query=expected_query, sort='updated', order='desc')

    # Check tqdm call using call_args
    mock_tqdm_class.assert_called_once()
    call_args, call_kwargs = mock_tqdm_class.call_args
    # The first positional argument should be the iterable
    assert call_args == () # No positional args expected
    assert call_kwargs == {'total': 1, 'desc': "Searching repos"}

    # Check update on the pbar instance
    mock_pbar.update.assert_called_once_with(1)

    assert len(results) == 1
    assert results[0] == mock_repository
    mock_sleep.assert_not_called()


@pytest.mark.unit
@patch('repobird_leadgen.github_search.tqdm') # Patch tqdm directly
@patch('repobird_leadgen.github_search.time.sleep')
@patch('builtins.print') # Mock print for cleaner test output
def test_search_readme_not_found(mock_print, mock_sleep, mock_tqdm_class, mock_auth, mock_github, mock_repository, mock_repo_data):
    # Configure the mock tqdm class
    mock_pbar = MagicMock()
    mock_pbar.update = MagicMock()
    mock_pbar.close = MagicMock()
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_context.__exit__.return_value = None
    mock_tqdm_class.return_value = mock_tqdm_context

    searcher = GitHubSearcher(token="dummy") # Uses mocks for init
    # searcher.gh is mock_github['instance']

    mock_repository.get_readme.side_effect = UnknownObjectException(status=404, data={}, headers={})
    mock_repos_list = [mock_repository]
    mock_github['search_results'].__iter__.return_value = iter(mock_repos_list)
    mock_github['search_results'].totalCount = 1

    results = searcher.search(label="test", language="any", min_stars=0, recent_days=365)

    mock_github['instance'].search_repositories.assert_called_once()

    # Check tqdm call using call_args
    mock_tqdm_class.assert_called_once()
    call_args, call_kwargs = mock_tqdm_class.call_args
    assert call_args == () # No positional args expected
    assert call_kwargs == {'total': 1, 'desc': "Searching repos"}

    # Check update on the pbar instance
    mock_pbar.update.assert_called_once_with(1) # Update is called for the item before get_readme fails

    assert len(results) == 1 # Search still returns the repo object
    assert results[0] == mock_repository
    mock_sleep.assert_not_called()


@pytest.mark.unit
@patch('repobird_leadgen.github_search.tqdm') # Patch tqdm directly
@patch('repobird_leadgen.github_search.time.sleep')
# @patch('builtins.print') # Temporarily removed for debugging
def test_search_rate_limit_during_iteration(mock_sleep, mock_tqdm_class, mock_auth, mock_github, mock_repository):
    # Configure the mock tqdm class
    mock_pbar = MagicMock()
    mock_pbar.update = MagicMock()
    mock_pbar.close = MagicMock()
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_context.__exit__.return_value = None
    mock_tqdm_class.return_value = mock_tqdm_context

    searcher = GitHubSearcher(token="dummy") # Uses mocks for init
    # searcher.gh is mock_github['instance']

    repo1 = mock_repository
    mock_iterator_retry = MagicMock()
    # Simulate: yield repo1, raise RateLimit, yield StopIteration after wait/retry
    # Make sure the exception has the attributes the SUT uses in the except block
    rate_limit_exception = RateLimitExceededException(status=403, data={}, headers={})
    mock_iterator_retry.__next__.side_effect = [repo1, rate_limit_exception, StopIteration]
    # Configure the iterable mock
    mock_github['search_results'].__iter__.return_value = mock_iterator_retry
    mock_github['search_results'].totalCount = 2

    # Adjust the reset time on the pre-configured mock rate limit object
    mock_github['rate_limit_info'].search.reset = datetime.now(timezone.utc) + timedelta(seconds=0.01)

    results = searcher.search(label="test", language="any", min_stars=0, recent_days=365, max_results=5)

    mock_github['instance'].search_repositories.assert_called_once()
    # Assert get_rate_limit was called *on the instance* after the exception
    mock_github['instance'].get_rate_limit.assert_called_once() # Use standard assertion
    mock_sleep.assert_called_once() # Sleep should be called

    # Check tqdm call using call_args
    mock_tqdm_class.assert_called_once()
    call_args, call_kwargs = mock_tqdm_class.call_args
    assert call_args == () # No positional args expected
    assert call_kwargs == {'total': 2, 'desc': "Searching repos"} # Total is 2

    # Check update on the pbar instance was called once for the first repo
    # The rate limit exception prevents the second update from occurring in the current logic
    mock_pbar.update.assert_called_once_with(1)

    assert len(results) == 1
    assert results[0] == repo1


@pytest.mark.unit
@patch('repobird_leadgen.github_search.tqdm') # Patch tqdm directly
def test_search_rate_limit_on_initial_search(mock_tqdm_class, mock_auth, mock_github):
    # No need to configure mock_tqdm_class return value as it shouldn't be called

    searcher = GitHubSearcher(token="dummy") # Uses mocks for init
    # searcher.gh is mock_github['instance']

    # Configure the side effect on the specific instance's method
    mock_github['instance'].search_repositories.side_effect = RateLimitExceededException(status=403, data={}, headers={})

    with pytest.raises(RateLimitExceededException):
        searcher.search(label="test", language="any", min_stars=0, recent_days=365)
    mock_github['instance'].search_repositories.assert_called_once()
    mock_tqdm_class.assert_not_called() # tqdm shouldn't be reached


@pytest.mark.unit
@patch('repobird_leadgen.github_search.tqdm') # Patch tqdm directly
@patch('repobird_leadgen.github_search.time.sleep')
# @patch('builtins.print') # Temporarily removed for debugging
def test_search_generic_github_exception_during_iteration(mock_sleep, mock_tqdm_class, mock_auth, mock_github, mock_repository):
    # Configure the mock tqdm class
    mock_pbar = MagicMock()
    mock_pbar.update = MagicMock()
    mock_pbar.close = MagicMock()
    mock_tqdm_context = MagicMock()
    mock_tqdm_context.__enter__.return_value = mock_pbar
    mock_tqdm_context.__exit__.return_value = None
    mock_tqdm_class.return_value = mock_tqdm_context

    searcher = GitHubSearcher(token="dummy") # Uses mocks for init
    # searcher.gh is mock_github['instance']

    repo1 = mock_repository
    mock_iterator = MagicMock()
    # Simulate: yield repo1, raise GithubException, yield StopIteration
    # Make sure the exception has the attributes the SUT uses in the except block
    generic_exception = GithubException(status=500, data={}, headers={})
    mock_iterator.__next__.side_effect = [repo1, generic_exception, StopIteration]
    # Configure the iterable mock
    mock_github['search_results'].__iter__.return_value = mock_iterator
    mock_github['search_results'].totalCount = 2

    results = searcher.search(label="test", language="any", min_stars=0, recent_days=365)

    mock_github['instance'].search_repositories.assert_called_once()

    # Check tqdm call using call_args
    mock_tqdm_class.assert_called_once()
    call_args, call_kwargs = mock_tqdm_class.call_args
    assert call_args == () # No positional args expected
    assert call_kwargs == {'total': 2, 'desc': "Searching repos"} # Total is 2

    # Check update on the pbar instance was called twice:
    # Once for the successful repo, once for the skipped repo
    assert mock_pbar.update.call_count == 2
    # Revert to simpler check first:
    # mock_pbar.update.assert_has_calls([call(1), call(1)]) # Check arguments too

    assert len(results) == 1 # SUT catches exception and continues, returns only repo1
    assert results[0] == repo1
    mock_sleep.assert_not_called()


# --- Integration Tests --- (No changes needed here)

@pytest.mark.integration
@skip_if_no_token
def test_basic_search_integration():
    gh = GitHubSearcher()
    try:
        repos = gh.search(label="good first issue", language="python", min_stars=1, recent_days=365 * 2, max_results=5)
        assert isinstance(repos, list)
        if repos:
            assert isinstance(repos[0], Repository)
        assert len(repos) <= 5
    except RateLimitExceededException:
         pytest.skip("GitHub API rate limit exceeded.")
    except GithubException as e:
        pytest.fail(f"GitHub API error during test_basic_search: {e.status} {e.data}")
    except Exception as e:
        pytest.fail(f"Unexpected error during test_basic_search: {e}")


@pytest.mark.integration
@skip_if_no_token
def test_good_first_issue_repos_helper_integration():
    gh = GitHubSearcher()
    try:
        repos = gh.good_first_issue_repos(language="python", min_stars=1, recent_days=365 * 2, max_results=3)
        assert isinstance(repos, list)
        if repos:
            assert isinstance(repos[0], Repository)
        assert len(repos) <= 3
    except RateLimitExceededException:
         pytest.skip("GitHub API rate limit exceeded.")
    except GithubException as e:
        pytest.fail(f"GitHub API error during test_good_first_issue_repos_helper: {e.status} {e.data}")
    except Exception as e:
        pytest.fail(f"Unexpected error during test_good_first_issue_repos_helper: {e}")

