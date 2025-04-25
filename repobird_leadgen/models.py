from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ContactInfo:
    emails: List[str] = field(default_factory=list)
    twitter: Optional[str] = None
    blog: Optional[str] = None


@dataclass
class RepoLead:
    """Represents a potential lead identified from a GitHub repository."""

    github_url: str
    repo_name: str
    description: Optional[str]
    stars: int
    forks: int
    owner_login: str
    owner_type: str  # e.g., 'User', 'Organization'
    last_pushed_at: datetime
    topics: List[str] = field(default_factory=list)
    readme_content: Optional[str] = None
    contacts: Optional[ContactInfo] = None  # Populated later by ContactScraper


@dataclass
class RepoSummary:
    """Represents a summarized view, possibly after contact scraping."""

    full_name: str  # e.g. "octocat/Hello‑World"
    description: str
    stars: int
    language: (
        str | None
    )  # Note: This wasn't in RepoLead, might need reconciliation later
    open_issues: int
    good_first_issues: int
    help_wanted_issues: int
    last_push: datetime
    contact: ContactInfo
