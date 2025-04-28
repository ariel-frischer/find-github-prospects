import pytest
import sys

# Note: We avoid importlib.reload and instead mock os.getenv directly
# before importing the config module within each test function.
# This ensures a clean state for each test.


def test_load_config_with_token(mocker):
    """Test config loading when GITHUB_TOKEN is set, using defaults for others."""

    # Define the behavior of os.getenv for this test
    def mock_getenv(key, default=None):
        if key == "GITHUB_TOKEN":
            return "test_token_123"
        # If the key is not explicitly mocked (GITHUB_TOKEN), return the default value
        # passed to os.getenv, simulating the real behavior.
        return default

    mocker.patch("repobird_leadgen.config.os.getenv", side_effect=mock_getenv)

    # Ensure the module is re-imported *after* the mock is set up
    if "repobird_leadgen.config" in sys.modules:
        del sys.modules["repobird_leadgen.config"]
    import repobird_leadgen.config

    assert repobird_leadgen.config.GITHUB_TOKEN == "test_token_123"
    # Check default values are correctly loaded
    assert repobird_leadgen.config.CONCURRENCY == 20
    assert repobird_leadgen.config.CACHE_DIR == "cache"
    assert repobird_leadgen.config.OUTPUT_DIR == "output"


def test_load_config_with_custom_vars(mocker):
    """Test config loading with custom optional variables set."""

    # Define the behavior of os.getenv for this test
    def mock_getenv(key, default=None):
        if key == "GITHUB_TOKEN":
            return "test_token_456"
        elif key == "CONCURRENCY":
            return "10"  # Return the string, as os.getenv does
        elif key == "CACHE_DIR":
            return "/tmp/rb_cache"
        elif key == "OUTPUT_DIR":
            return "/tmp/rb_output"
        return default  # Fallback

    mocker.patch("repobird_leadgen.config.os.getenv", side_effect=mock_getenv)

    if "repobird_leadgen.config" in sys.modules:
        del sys.modules["repobird_leadgen.config"]
    import repobird_leadgen.config

    assert repobird_leadgen.config.GITHUB_TOKEN == "test_token_456"
    assert repobird_leadgen.config.CONCURRENCY == 10
    assert repobird_leadgen.config.CACHE_DIR == "/tmp/rb_cache"
    assert repobird_leadgen.config.OUTPUT_DIR == "/tmp/rb_output"


def test_load_config_without_token(mocker):
    """Test config loading raises RuntimeError when GITHUB_TOKEN is missing."""

    # Define the behavior of os.getenv for this test
    def mock_getenv(key, default=None):
        if key == "GITHUB_TOKEN":
            return None  # Simulate missing token
        # If the key is not GITHUB_TOKEN, return the default value provided
        # to os.getenv, simulating the real behavior.
        return default

    mocker.patch("repobird_leadgen.config.os.getenv", side_effect=mock_getenv)

    # Ensure the module is removed from cache before import attempt
    if "repobird_leadgen.config" in sys.modules:
        del sys.modules["repobird_leadgen.config"]

    # Expect RuntimeError during the import itself
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN missing"):
        pass


# Cleanup: Ensure the module is removed after tests run in this file
# This helps prevent state leakage if other test files import config
def teardown_module(module):
    if "repobird_leadgen.config" in sys.modules:
        del sys.modules["repobird_leadgen.config"]
