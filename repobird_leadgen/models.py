from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class ContactInfo:
    emails: List[str] = field(default_factory=list)
    twitter: Optional[str] = None
    blog: Optional[str] = None

@dataclass
class RepoSummary:
    full_name: str                # e.g. "octocat/Hello‑World"
    description: str
    stars: int
    language: str | None
    open_issues: int
    good_first_issues: int
    help_wanted_issues: int
    last_push: datetime
    contact: ContactInfo
