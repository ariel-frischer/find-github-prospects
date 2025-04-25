"""Tests for repobird_leadgen.summarizer"""

import pytest
import json
import csv
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, call
from typing import List  # Added import for List

from repobird_leadgen.models import RepoSummary, ContactInfo
from repobird_leadgen.summarizer import (
    to_markdown,
    to_jsonl,
    to_csv,
    save_summary,
    _format_output_path,
)

# --- Test Data Fixtures ---


@pytest.fixture
def sample_contact_info_basic():
    """Provides a basic ContactInfo object."""
    return ContactInfo(emails=["test@example.com"])


@pytest.fixture
def sample_contact_info_full():
    """Provides a full ContactInfo object."""
    return ContactInfo(
        emails=["john.doe@example.com", "jdoe@work.com"],
        twitter="johndoe",
        blog="https://johndoe.blog",
    )


@pytest.fixture
def sample_repo_summary_basic(sample_contact_info_basic):
    """Provides a basic RepoSummary object."""
    return RepoSummary(
        full_name="user/basic_repo",
        description="A basic repository.",
        stars=10,
        language="Python",
        open_issues=5,
        good_first_issues=1,
        help_wanted_issues=0,
        last_push=datetime.fromisoformat("2024-01-01T00:00:00"),
        contact=sample_contact_info_basic,
    )


@pytest.fixture
def sample_repo_summary_full(sample_contact_info_full):
    """Provides a RepoSummary object with more details."""
    return RepoSummary(
        full_name="user/full_repo",
        description="A full repository description.",
        stars=100,
        language=None,  # Test None language
        open_issues=20,
        good_first_issues=5,
        help_wanted_issues=2,
        last_push=datetime.fromisoformat("2024-04-25T12:00:00"),
        contact=sample_contact_info_full,
    )


@pytest.fixture
def sample_summaries(sample_repo_summary_basic, sample_repo_summary_full):
    """Provides a list of sample RepoSummary objects."""
    return [sample_repo_summary_basic, sample_repo_summary_full]


# --- Helper for checking Markdown --- Needed because Rich adds styling
def extract_markdown_table_content(md_string: str) -> List[List[str]]:
    """Extracts rows from a simple Markdown table."""
    lines = md_string.strip().split("\n")
    # Skip title, header separator
    table_lines = [line for line in lines if line.startswith("|")][
        1:
    ]  # Skip header line itself
    if len(table_lines) > 0 and "---" in table_lines[0]:
        table_lines = table_lines[1:]  # Skip separator line

    data = []
    for line in table_lines:
        cols = [col.strip() for col in line.strip("|").split("|")]
        data.append(cols)
    return data


# --- Tests for Formatting Functions ---
# Note: to_markdown now primarily prints to console and saves a file.
# We will test the file writing part here, assuming console output is handled by Rich.


def test_to_markdown_file_content(sample_summaries, tmp_path):
    """Test the content written to the markdown file."""
    outfile = tmp_path / "test.md"
    # Suppress console output during test
    with patch(
        "repobird_leadgen.summarizer.console.print"
    ) as _:  # Changed mock_print to _
        to_markdown(sample_summaries, outfile)

    assert outfile.exists()
    content = outfile.read_text()

    assert "# RepoBird LeadGen Prospects (test)" in content
    assert (
        "| Repo | Stars | Lang | Open Issues | Good First | Help Wanted | Last Push | Emails | Twitter | Blog |"
        in content
    )
    assert "|---|---|---|---|---|---|---|---|---|---|" in content

    table_data = extract_markdown_table_content(content)
    assert len(table_data) == 2
    # Basic repo
    assert "[user/basic_repo](https://github.com/user/basic_repo)" in table_data[0][0]
    assert table_data[0][1] == "10"
    assert table_data[0][2] == "Python"
    assert table_data[0][7] == "`test@example.com`"
    assert table_data[0][8] == "-"
    assert table_data[0][9] == "-"
    # Full repo
    assert "[user/full_repo](https://github.com/user/full_repo)" in table_data[1][0]
    assert table_data[1][1] == "100"
    assert table_data[1][2] == "-"  # Language is None
    assert table_data[1][7] == "`john.doe@example.com, jdoe@work.com`"
    assert "[@johndoe](https://twitter.com/johndoe)" in table_data[1][8]
    assert "[Blog](https://johndoe.blog)" in table_data[1][9]


def test_to_jsonl_content(sample_summaries, tmp_path):
    """Test the content written to the jsonl file."""
    outfile = tmp_path / "test.jsonl"
    with patch(
        "repobird_leadgen.summarizer.console.print"
    ) as _:  # Changed mock_print to _
        to_jsonl(sample_summaries, outfile)

    assert outfile.exists()
    lines = outfile.read_text().strip().split("\n")
    assert len(lines) == 2

    data1 = json.loads(lines[0])
    data2 = json.loads(lines[1])

    assert data1["full_name"] == "user/basic_repo"
    assert data1["language"] == "Python"
    assert data1["last_push"] == "2024-01-01T00:00:00"
    assert data1["contact"]["emails"] == ["test@example.com"]
    assert data1["contact"]["twitter"] is None

    assert data2["full_name"] == "user/full_repo"
    assert data2["language"] is None
    assert data2["last_push"] == "2024-04-25T12:00:00"
    assert data2["contact"]["emails"] == ["john.doe@example.com", "jdoe@work.com"]
    assert data2["contact"]["twitter"] == "johndoe"
    assert data2["contact"]["blog"] == "https://johndoe.blog"


def test_to_csv_content(sample_summaries, tmp_path):
    """Test the content written to the csv file."""
    outfile = tmp_path / "test.csv"
    with patch(
        "repobird_leadgen.summarizer.console.print"
    ) as _:  # Changed mock_print to _
        to_csv(sample_summaries, outfile)

    assert outfile.exists()
    with outfile.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    expected_header = [
        "full_name",
        "description",
        "stars",
        "language",
        "open_issues",
        "good_first_issues",
        "help_wanted_issues",
        "last_push",
        "contact_emails",
        "contact_twitter",
        "contact_blog",
    ]
    assert header == expected_header
    assert len(rows) == 2

    # Basic repo
    assert rows[0][0] == "user/basic_repo"
    assert rows[0][3] == "Python"
    assert rows[0][7] == "2024-01-01 00:00:00"
    assert rows[0][8] == "test@example.com"
    assert rows[0][9] == ""
    assert rows[0][10] == ""

    # Full repo
    assert rows[1][0] == "user/full_repo"
    assert rows[1][3] == ""  # Language is None
    assert rows[1][7] == "2024-04-25 12:00:00"
    assert rows[1][8] == "john.doe@example.com,jdoe@work.com"
    assert rows[1][9] == "johndoe"
    assert rows[1][10] == "https://johndoe.blog"


# --- Tests for save_summary ---


@patch("repobird_leadgen.summarizer.to_markdown")
@patch("repobird_leadgen.summarizer._format_output_path")
def test_save_summary_markdown(
    mock_format_path, mock_to_markdown, sample_summaries, tmp_path
):
    """Test save_summary calls to_markdown correctly."""
    outdir_base = str(tmp_path / "output")
    mock_md_path = tmp_path / "output" / "prospects_ts.md"
    mock_format_path.return_value = mock_md_path

    with patch(
        "repobird_leadgen.summarizer.console.print"
    ) as _:  # Changed mock_print to _
        save_summary(sample_summaries, outdir_base, ["md"])

    mock_format_path.assert_called_once_with(Path(outdir_base), "prospects", "md")
    mock_to_markdown.assert_called_once_with(sample_summaries, mock_md_path)


@patch("repobird_leadgen.summarizer.to_jsonl")
@patch("repobird_leadgen.summarizer._format_output_path")
def test_save_summary_jsonl(
    mock_format_path, mock_to_jsonl, sample_summaries, tmp_path
):
    """Test save_summary calls to_jsonl correctly."""
    outdir_base = str(tmp_path / "output")
    mock_jsonl_path = tmp_path / "output" / "prospects_ts.jsonl"
    mock_format_path.return_value = mock_jsonl_path

    with patch(
        "repobird_leadgen.summarizer.console.print"
    ) as _:  # Changed mock_print to _
        save_summary(sample_summaries, outdir_base, ["jsonl"])

    mock_format_path.assert_called_once_with(Path(outdir_base), "prospects", "jsonl")
    mock_to_jsonl.assert_called_once_with(sample_summaries, mock_jsonl_path)


@patch("repobird_leadgen.summarizer.to_csv")
@patch("repobird_leadgen.summarizer._format_output_path")
def test_save_summary_csv(mock_format_path, mock_to_csv, sample_summaries, tmp_path):
    """Test save_summary calls to_csv correctly."""
    outdir_base = str(tmp_path / "output")
    mock_csv_path = tmp_path / "output" / "prospects_ts.csv"
    mock_format_path.return_value = mock_csv_path

    with patch(
        "repobird_leadgen.summarizer.console.print"
    ) as _:  # Changed mock_print to _
        save_summary(sample_summaries, outdir_base, ["csv"])

    mock_format_path.assert_called_once_with(Path(outdir_base), "prospects", "csv")
    mock_to_csv.assert_called_once_with(sample_summaries, mock_csv_path)


@patch("repobird_leadgen.summarizer.to_markdown")
@patch("repobird_leadgen.summarizer.to_jsonl")
@patch("repobird_leadgen.summarizer.to_csv")
@patch("repobird_leadgen.summarizer._format_output_path")
def test_save_summary_multiple_formats(
    mock_format_path,
    mock_to_csv,
    mock_to_jsonl,
    mock_to_markdown,
    sample_summaries,
    tmp_path,
):
    """Test save_summary calls all formatters when multiple formats are requested."""
    outdir_base = str(tmp_path / "output")
    # Define paths for each format
    mock_md_path = tmp_path / "output" / "prospects_ts.md"
    mock_jsonl_path = tmp_path / "output" / "prospects_ts.jsonl"
    mock_csv_path = tmp_path / "output" / "prospects_ts.csv"

    # Mock _format_output_path to return the correct path based on extension
    def side_effect(outdir, base, ext):
        if ext == "md":
            return mock_md_path
        if ext == "jsonl":
            return mock_jsonl_path
        if ext == "csv":
            return mock_csv_path
        raise ValueError("Unexpected format")

    mock_format_path.side_effect = side_effect

    with patch(
        "repobird_leadgen.summarizer.console.print"
    ) as _:  # Changed mock_print to _
        save_summary(sample_summaries, outdir_base, ["md", "jsonl", "csv"])

    # Check _format_output_path calls
    format_calls = [
        call(Path(outdir_base), "prospects", "md"),
        call(Path(outdir_base), "prospects", "jsonl"),
        call(Path(outdir_base), "prospects", "csv"),
    ]
    mock_format_path.assert_has_calls(format_calls, any_order=True)
    assert mock_format_path.call_count == 3

    # Check formatter calls
    mock_to_markdown.assert_called_once_with(sample_summaries, mock_md_path)
    mock_to_jsonl.assert_called_once_with(sample_summaries, mock_jsonl_path)
    mock_to_csv.assert_called_once_with(sample_summaries, mock_csv_path)


def test_save_summary_no_summaries(tmp_path):
    """Test save_summary does nothing when summaries list is empty."""
    outdir_base = str(tmp_path / "output")
    with (
        patch("repobird_leadgen.summarizer.to_markdown") as mock_md,
        patch("repobird_leadgen.summarizer.to_jsonl") as mock_jsonl,
        patch("repobird_leadgen.summarizer.to_csv") as mock_csv,
        patch("repobird_leadgen.summarizer._format_output_path") as mock_format,
        patch("repobird_leadgen.summarizer.console.print") as mock_console_print,
    ):  # Assign to named var
        save_summary([], outdir_base, ["md", "jsonl", "csv"])

    mock_format.assert_not_called()
    mock_md.assert_not_called()
    mock_jsonl.assert_not_called()
    mock_csv.assert_not_called()
    mock_console_print.assert_called_with(
        "[yellow]No summaries generated, skipping output file creation."
    )  # Use named var


def test_save_summary_invalid_format_ignored(sample_summaries, tmp_path):
    """Test save_summary ignores invalid formats in the list."""
    outdir_base = str(tmp_path / "output")
    with (
        patch("repobird_leadgen.summarizer.to_markdown") as mock_md,
        patch("repobird_leadgen.summarizer.to_jsonl") as mock_jsonl,
        patch("repobird_leadgen.summarizer.to_csv") as mock_csv,
        patch("repobird_leadgen.summarizer._format_output_path") as mock_format,
        patch("repobird_leadgen.summarizer.console.print") as _,
    ):  # Changed mock_print to _
        mock_format.return_value = tmp_path / "output" / "prospects_ts.md"
        save_summary(
            sample_summaries, outdir_base, ["md", "invalid", ""]
        )  # Include invalid/empty

    mock_format.assert_called_once_with(Path(outdir_base), "prospects", "md")
    mock_md.assert_called_once()
    mock_jsonl.assert_not_called()
    mock_csv.assert_not_called()


# --- Test _format_output_path ---


@patch("repobird_leadgen.summarizer.datetime")
def test_format_output_path(mock_datetime, tmp_path):
    """Test the output path formatting function."""
    mock_now = datetime(2024, 4, 25, 10, 30, 0)
    mock_datetime.utcnow.return_value = mock_now

    outdir = tmp_path / "test_out"
    base_name = "results"
    extension = "txt"

    expected_path = outdir / "results_2024-04-25_103000.txt"
    actual_path = _format_output_path(outdir, base_name, extension)

    assert actual_path == expected_path
    assert outdir.exists()  # Check directory creation
    mock_datetime.utcnow.assert_called_once()
