import importlib
import logging
import sys

import pytest

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

    # 1. Prevent load_dotenv from running in this test
    mocker.patch("repobird_leadgen.config.load_dotenv", return_value=None)

    # 2. Mock os.getenv to return None for GITHUB_TOKEN
    def mock_getenv(key, default=None):
        if key == "GITHUB_TOKEN":
            return None  # Simulate missing token
        # If the key is not GITHUB_TOKEN, return the default value provided
        # to os.getenv, simulating the real behavior.
        return default

    mocker.patch("repobird_leadgen.config.os.getenv", side_effect=mock_getenv)

    # 3. Ensure the module is removed from cache before import attempt
    # 3. Ensure the module is removed from cache before import attempt
    config_module_name = "repobird_leadgen.config"
    if config_module_name in sys.modules:
        logging.warning(f"Removing '{config_module_name}' from sys.modules")
        del sys.modules[config_module_name]
    else:
        logging.info(f"'{config_module_name}' not found in sys.modules. Good.")

    # Try calling the mocked getenv to see what it returns for the token
    import os

    logging.info(
        f"Mocked os.getenv('GITHUB_TOKEN') returns: {os.getenv('GITHUB_TOKEN')}"
    )

    # Expect RuntimeError during the import of the config module itself,
    # which happens when github_search (or config directly) is imported.
    # 4. Expect RuntimeError during import
    logging.info(f"Attempting import of '{config_module_name}' inside pytest.raises...")
    try:
        with pytest.raises(
            RuntimeError, match="GITHUB_TOKEN missing – create a PAT and add to .env"
        ):
            # Use import_module to explicitly trigger the import within the context
            importlib.import_module(config_module_name)
        logging.info("Successfully caught expected RuntimeError on import.")
    except Exception as e:
        logging.error(
            f"Caught unexpected exception during import attempt: {e}", exc_info=True
        )
        pytest.fail(f"Import raised unexpected exception: {e}")
    # This part will only be reached if pytest.raises fails
    logging.error(
        f"FAILED: Did not raise RuntimeError during import of '{config_module_name}'"
    )


# Cleanup: Ensure the module is removed after tests run in this file
# This helps prevent state leakage if other test files import config
def teardown_module(module):
    if "repobird_leadgen.config" in sys.modules:
        del sys.modules["repobird_leadgen.config"]
