"""Tests for repobird_leadgen.summarizer"""

import json
from unittest.mock import MagicMock, patch

import pytest

from repobird_leadgen.summarizer import save_enriched_data

# --- Test Data Fixtures ---


@pytest.fixture
def sample_enriched_data_basic():
    """Provides a basic list of enriched data dictionaries."""
    return [
        {
            "original_repo_data": {"full_name": "user/repo1", "stars": 10},
            "issue_analysis": [
                {
                    "issue_number": 1,
                    "issue_url": "url1",
                    "complexity_score": 0.2,
                    "is_good_first_issue_for_agent": True,
                    "reasoning": "Simple fix",
                    "required_domain_knowledge": 0.1,
                    "codebase_familiarity_needed": 0.0,
                    "scope_clarity": 0.9,
                    "meta_notes": None,
                    "error": None,
                }
            ],
            "enrichment_error": None,
        }
    ]


@pytest.fixture
def sample_enriched_data_multiple():
    """Provides a list with multiple enriched data dictionaries."""
    return [
        {
            "original_repo_data": {"full_name": "user/repo1", "stars": 10},
            "issue_analysis": [
                {
                    "issue_number": 1,
                    "issue_url": "url1",
                    "complexity_score": 0.2,
                    "is_good_first_issue_for_agent": True,
                    "reasoning": "Simple fix",
                    "required_domain_knowledge": 0.1,
                    "codebase_familiarity_needed": 0.0,
                    "scope_clarity": 0.9,
                    "meta_notes": None,
                    "error": None,
                }
            ],
            "enrichment_error": None,
        },
        {
            "original_repo_data": {"full_name": "user/repo2", "stars": 50},
            "issue_analysis": [
                {
                    "issue_number": 5,
                    "issue_url": "url5",
                    "complexity_score": 0.8,
                    "is_good_first_issue_for_agent": False,
                    "reasoning": "Complex refactor",
                    "required_domain_knowledge": 0.7,
                    "codebase_familiarity_needed": 0.8,
                    "scope_clarity": 0.5,
                    "meta_notes": "Needs discussion",
                    "error": None,
                },
                {
                    "issue_number": 6,
                    "issue_url": "url6",
                    "complexity_score": None,
                    "is_good_first_issue_for_agent": None,
                    "reasoning": None,
                    "required_domain_knowledge": None,
                    "codebase_familiarity_needed": None,
                    "scope_clarity": None,
                    "meta_notes": None,
                    "error": "LLM analysis failed",
                },
            ],
            "enrichment_error": None,
        },
        {
            "original_repo_data": {"full_name": "user/repo3", "stars": 5},
            "issue_analysis": [],
            "enrichment_error": "Could not process repo",
        },
    ]


# --- Tests for save_enriched_data ---


def test_save_enriched_data_creates_file_and_writes_jsonl(
    sample_enriched_data_multiple, tmp_path
):
    """Test that the function creates the output file and writes correct JSONL."""
    outfile = tmp_path / "output" / "enriched_test.jsonl"
    with patch("repobird_leadgen.summarizer.console.print") as mock_print:
        save_enriched_data(sample_enriched_data_multiple, outfile)

    assert outfile.exists()
    assert outfile.parent.exists()  # Check directory creation

    lines = outfile.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(sample_enriched_data_multiple)

    # Verify content of each line
    for i, line in enumerate(lines):
        loaded_data = json.loads(line)
        assert loaded_data == sample_enriched_data_multiple[i]

    # Check console output
    mock_print.assert_called_with(f"[green]Saved enriched data (JSONL) → {outfile}")


def test_save_enriched_data_empty_list(tmp_path):
    """Test that the function handles an empty input list gracefully."""
    outfile = tmp_path / "output" / "empty_enriched.jsonl"
    with patch("repobird_leadgen.summarizer.console.print") as mock_print:
        save_enriched_data([], outfile)

    assert not outfile.exists()  # File should not be created
    mock_print.assert_called_with(
        "[yellow]No enriched data provided, skipping output file creation."
    )


@patch("pathlib.Path.open")
def test_save_enriched_data_handles_write_error(mock_open, tmp_path):
    """Test that the function catches and reports errors during file writing."""
    outfile = tmp_path / "output" / "error_enriched.jsonl"
    mock_open.side_effect = OSError("Disk full")  # Simulate a write error

    with patch("repobird_leadgen.summarizer.console.print") as mock_print:
        # Use a minimal valid data list to trigger the write attempt
        save_enriched_data([{"original_repo_data": {}, "issue_analysis": []}], outfile)

    # Check that the error was printed to the console
    mock_print.assert_any_call(
        f"[red]Error writing enriched JSONL file {outfile}: Disk full"
    )
    # Ensure the success message was NOT printed
    success_call = MagicMock(
        return_value=f"[green]Saved enriched data (JSONL) → {outfile}"
    )
    assert success_call not in mock_print.call_args_list
