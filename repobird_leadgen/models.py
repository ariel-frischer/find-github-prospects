from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# Define the Pydantic model for LLM output validation
class LLMIssueAnalysisSchema(BaseModel):
    """
    Pydantic model defining the structured JSON output expected from the LLM
    after analyzing a GitHub issue. It includes various metrics and a detailed
    problem statement synthesis.
    """

    full_problem_statement: str = Field(
        ...,
        description="A comprehensive synthesis of the issue. Combine the title, body, and relevant comment details into a clear, self-contained problem description suitable for an AI agent to understand the task without needing to re-read the entire original issue thread. Focus on the core task, requirements, and any constraints mentioned.",
    )
    complexity_score: float = Field(
        ...,
        description="A numerical score from 0.0 (very simple, e.g., typo fix, documentation update) to 1.0 (very complex, e.g., major refactoring, new core feature, requires deep architectural understanding) representing the estimated difficulty of resolving the issue.",
    )
    is_good_first_issue_for_agent: bool = Field(
        ...,
        description="A boolean flag indicating if this issue is suitable for an AI software agent to attempt as a relatively standalone task. Consider factors like clarity, scope, required external interactions, and complexity.",
    )
    reasoning: str = Field(
        ...,
        description="A concise explanation justifying the `complexity_score` and `is_good_first_issue_for_agent` assessment. Mention key factors like ambiguity, required knowledge, or specific challenges.",
    )
    required_domain_knowledge: float = Field(
        ...,
        description="A numerical score from 0.0 (general programming knowledge sufficient) to 1.0 (requires deep, specialized domain expertise specific to the project or its field) indicating the level of domain-specific understanding needed.",
    )
    codebase_familiarity_needed: float = Field(
        ...,
        description="A numerical score from 0.0 (can be solved by modifying isolated files/functions) to 1.0 (requires understanding broad sections of the codebase, architecture, and interactions between components) indicating how much of the existing codebase needs to be understood.",
    )
    scope_clarity: float = Field(
        ...,
        description="A numerical score from 0.0 (very ambiguous, unclear requirements or acceptance criteria) to 1.0 (very clearly defined scope, specific requirements, and measurable outcomes) indicating how well-defined the issue's scope is.",
    )
    meta_notes: str = Field(
        ...,
        description="Concise string for any other critical observations not captured elsewhere. Note potential blockers (e.g., needs external API key, depends on unavailable service), required human interaction (e.g., needs design decision, requires user feedback), significant ambiguity, or missing information.",
    )
    readme_summary: Optional[str] = Field(
        None,
        description="A brief summary of the repository's README content, focusing on aspects relevant to understanding the project's purpose and potentially the context of the issue. This field should only be populated if README content was successfully scraped and provided.",
    )

    # Optional: Add a validator if needed, though LiteLLM handles schema enforcement
    # @field_validator('readme_summary')
    # def check_readme_summary(cls, v, values):
    #     # Example: Ensure readme_summary is None if no readme content was provided
    #     # This logic might be better handled during prompt construction or data processing
    #     pass


# Pydantic model for URL Summarization LLM output
class UrlSummarySchema(BaseModel):
    """
    Pydantic model defining the structured JSON output expected from the LLM
    after analyzing content scraped from a URL linked in a GitHub issue.
    """

    is_relevant: bool = Field(
        ...,
        description="Boolean flag indicating if the content from the URL is relevant to the original GitHub issue's title and body.",
    )
    summary: Optional[str] = Field(
        None,
        description="A concise summary of the URL's content, focusing *only* on aspects relevant to the original GitHub issue. Should be null or empty if is_relevant is false.",
    )

    @field_validator("summary")
    def check_summary_relevance(
        cls, v: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        """Ensure summary is empty or None if is_relevant is False."""
        # Need to access 'is_relevant' which might not be validated yet.
        # Pydantic v2 runs validators based on definition order or explicitly.
        # Let's access the raw data if possible, or rely on the description.
        # A simpler approach is to handle this *after* validation if needed.
        # For now, we trust the LLM prompt and LiteLLM's enforcement.
        # If strict enforcement is needed post-LLM:
        data = (
            values.data
        )  # Access raw data before validation completes for other fields
        if "is_relevant" in data and not data["is_relevant"]:
            if v is not None and v.strip() != "":
                # Optionally raise ValueError or just return None/empty string
                # raise ValueError("Summary must be empty if is_relevant is False")
                return None  # Or ""
        return v


# Define the internal data structure, including fields not generated by LLM
class IssueAnalysis(LLMIssueAnalysisSchema):
    """
    Internal representation of issue analysis results.
    Includes LLM-generated fields (validated by parent) plus identifying info and potential errors.
    """

    issue_number: int = Field(..., description="The GitHub issue number.")
    issue_url: str = Field(..., description="The full URL to the GitHub issue.")
    # Allow error field to be optional, as it's for processing errors, not LLM output
    error: Optional[str] = Field(
        None, description="Stores errors encountered during scraping or analysis."
    )


@dataclass
class EnrichedRepoData:
    """
    Represents the data for a repository after issue analysis.
    Includes the original data from the search cache plus the analysis results.
    """

    # Fields from the original search cache JSONL
    # Using Dict[str, Any] for flexibility as cache format might evolve slightly
    original_repo_data: Dict[str, Any] = field(default_factory=dict)

    # Analysis results for the issues found
    issue_analysis: List[IssueAnalysis] = field(default_factory=list)

    # Optional field to indicate errors during the enrichment process itself
    enrichment_error: Optional[str] = None
