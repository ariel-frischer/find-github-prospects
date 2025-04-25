import pytest
import os
import importlib

# Ensure the module is loaded fresh for each test if needed
# This is crucial because config is loaded at import time


def test_load_config_with_token(mocker):
    """Test config loading when GITHUB_TOKEN is set."""
    # Set the environment variable using mocker
    mocker.patch.dict(os.environ, {"GITHUB_TOKEN": "test_token_123"})

    # Reload the config module to pick up the mocked environment variable
    # Use a local import to avoid polluting the global namespace if module state is tricky
    import repobird_leadgen.config

    importlib.reload(repobird_leadgen.config)

    assert repobird_leadgen.config.GITHUB_TOKEN == "test_token_123"
    # Check default values
    assert repobird_leadgen.config.CONCURRENCY == 20
    assert repobird_leadgen.config.CACHE_DIR == "cache"
    assert repobird_leadgen.config.OUTPUT_DIR == "output"


def test_load_config_with_custom_vars(mocker):
    """Test config loading with custom optional variables set."""
    mocker.patch.dict(
        os.environ,
        {
            "GITHUB_TOKEN": "test_token_456",
            "CONCURRENCY": "10",
            "CACHE_DIR": "/tmp/rb_cache",
            "OUTPUT_DIR": "/tmp/rb_output",
        },
    )

    import repobird_leadgen.config

    importlib.reload(repobird_leadgen.config)

    assert repobird_leadgen.config.GITHUB_TOKEN == "test_token_456"
    assert repobird_leadgen.config.CONCURRENCY == 10
    assert repobird_leadgen.config.CACHE_DIR == "/tmp/rb_cache"
    assert repobird_leadgen.config.OUTPUT_DIR == "/tmp/rb_output"


def test_load_config_without_token(mocker):
    """Test config loading raises RuntimeError when GITHUB_TOKEN is missing."""
    # Ensure GITHUB_TOKEN is not in the environment
    # Clear os.environ and patch it to be empty
    mocker.patch.dict(os.environ, {}, clear=True)

    # Expect RuntimeError during module import/reload
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN missing"):
        # Need to reload the module to trigger the check again
        import repobird_leadgen.config

        # We need to ensure the module is *actually* reloaded,
        # sometimes Python caches imports aggressively.
        # Using a fresh import inside the test or reload is key.
        importlib.reload(repobird_leadgen.config)


# Cleanup environment variables after tests if necessary, although mocker should handle this.
# A fixture could also be used to manage the config module state.
