"""Tests for repobird_leadgen.models"""

import pytest
from pydantic import ValidationError

from repobird_leadgen.models import EnrichedRepoData, IssueAnalysis

# --- Tests for IssueAnalysis (Pydantic Model) ---

# Minimal valid data for required fields (excluding error)
# Added 'full_problem_statement'
VALID_ANALYSIS_DATA = {
    "full_problem_statement": "The user wants to update the README file to include installation instructions.",
    "complexity_score": 0.5,
    "is_good_first_issue_for_agent": True,
    "reasoning": "Seems straightforward.",
    "required_domain_knowledge": 0.2,
    "codebase_familiarity_needed": 0.1,
    "scope_clarity": 0.9,
    "meta_notes": "Needs tests.",
}


def test_issue_analysis_instantiation_valid():
    """Test IssueAnalysis instantiation with all required fields."""
    data = {
        "issue_number": 456,
        "issue_url": "https://github.com/test/repo/issues/456",
        **VALID_ANALYSIS_DATA,
        "error": "Some minor warning.",  # Optional error field
    }
    analysis = IssueAnalysis(**data)

    assert analysis.issue_number == 456
    assert analysis.issue_url == "https://github.com/test/repo/issues/456"
    assert analysis.complexity_score == VALID_ANALYSIS_DATA["complexity_score"]
    assert (
        analysis.is_good_first_issue_for_agent
        == VALID_ANALYSIS_DATA["is_good_first_issue_for_agent"]
    )
    assert analysis.reasoning == VALID_ANALYSIS_DATA["reasoning"]
    assert (
        analysis.required_domain_knowledge
        == VALID_ANALYSIS_DATA["required_domain_knowledge"]
    )
    assert (
        analysis.codebase_familiarity_needed
        == VALID_ANALYSIS_DATA["codebase_familiarity_needed"]
    )
    assert analysis.scope_clarity == VALID_ANALYSIS_DATA["scope_clarity"]
    assert analysis.meta_notes == VALID_ANALYSIS_DATA["meta_notes"]
    assert analysis.error == "Some minor warning."  # Check optional field


def test_issue_analysis_instantiation_valid_no_error():
    """Test IssueAnalysis instantiation with required fields and no error."""
    data = {
        "issue_number": 123,
        "issue_url": "https://github.com/test/repo/issues/123",
        **VALID_ANALYSIS_DATA,
        # No error field provided
    }
    analysis = IssueAnalysis(**data)

    assert analysis.issue_number == 123
    assert analysis.issue_url == "https://github.com/test/repo/issues/123"
    assert analysis.complexity_score == VALID_ANALYSIS_DATA["complexity_score"]
    assert analysis.error is None  # Default for optional field


def test_issue_analysis_instantiation_missing_required_field():
    """Test Pydantic raises ValidationError if a required field is missing."""
    invalid_data = {
        "issue_number": 789,
        "issue_url": "https://github.com/test/repo/issues/789",
        # Missing 'complexity_score' and others from VALID_ANALYSIS_DATA
        "reasoning": "Incomplete data.",
    }
    with pytest.raises(ValidationError):
        IssueAnalysis(**invalid_data)


def test_issue_analysis_instantiation_incorrect_type():
    """Test Pydantic raises ValidationError if a field has the wrong type."""
    invalid_data = {
        "issue_number": 789,
        "issue_url": "https://github.com/test/repo/issues/789",
        **VALID_ANALYSIS_DATA,
        "complexity_score": "not-a-float",  # Incorrect type
    }
    with pytest.raises(ValidationError):
        IssueAnalysis(**invalid_data)


# --- Tests for EnrichedRepoData (remains a dataclass) ---
# Note: EnrichedRepoData holds IssueAnalysis *objects* internally,
# but the tests below check its instantiation with data that *would*
# create those objects. The structure passed to EnrichedRepoData constructor
# in tests might need adjustment if it expects IssueAnalysis objects directly.
# However, the current tests seem to instantiate IssueAnalysis separately
# and pass the object, which should still work. Let's verify.


def test_enriched_repo_data_instantiation_defaults():
    """Test EnrichedRepoData instantiation with default values."""
    enriched_data = EnrichedRepoData()
    assert enriched_data.original_repo_data == {}
    assert enriched_data.issue_analysis == []
    assert enriched_data.enrichment_error is None


def test_enriched_repo_data_instantiation_with_pydantic_models():
    """Test EnrichedRepoData instantiation with Pydantic IssueAnalysis objects."""
    original_data = {"full_name": "test/repo", "stars": 100}

    # Create valid Pydantic IssueAnalysis instances
    analysis1_data = {
        "issue_number": 1,
        "issue_url": "https://github.com/test/repo/issues/1",
        **VALID_ANALYSIS_DATA,
    }
    analysis1 = IssueAnalysis(**analysis1_data)

    analysis2_data = {
        "issue_number": 2,
        "issue_url": "https://github.com/test/repo/issues/2",
        **VALID_ANALYSIS_DATA,
        "error": "Analysis failed",  # Add optional error
    }
    # Modify one field for variety
    analysis2_data["reasoning"] = "Different reason"
    analysis2 = IssueAnalysis(**analysis2_data)

    issue_analyses_list: List[IssueAnalysis] = [analysis1, analysis2]
    error_msg = "Top-level enrichment error"

    # Instantiate the dataclass with the list of Pydantic models
    enriched_data = EnrichedRepoData(
        original_repo_data=original_data,
        issue_analysis=issue_analyses_list,  # Correct variable name
        enrichment_error=error_msg,
    )

    # Assertions remain largely the same, checking the contained objects
    assert enriched_data.original_repo_data == original_data
    assert enriched_data.issue_analysis == issue_analyses_list
    assert enriched_data.enrichment_error == error_msg
    # Access attributes of the Pydantic models within the list
    assert enriched_data.issue_analysis[0].issue_number == 1
    assert enriched_data.issue_analysis[0].reasoning == VALID_ANALYSIS_DATA["reasoning"]
    assert enriched_data.issue_analysis[1].issue_number == 2
    assert enriched_data.issue_analysis[1].reasoning == "Different reason"
    assert enriched_data.issue_analysis[1].error == "Analysis failed"
