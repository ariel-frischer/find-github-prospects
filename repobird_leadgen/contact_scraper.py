import re
from pathlib import Path
from typing import List, Optional, Set
from github import GithubException
from github.Repository import Repository
from github.ContentFile import ContentFile
from github.NamedUser import NamedUser
from .models import ContactInfo
from rich import print
import github # Make sure github module is imported for GithubException

# More robust email regex (RFC 5322 simplified)
_EMAIL_RE = re.compile(r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[--!#-[]-]|\[-	-])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[--!-ZS-]|\[-	-])+)\])""", re.IGNORECASE)
_TWITTER_RE = re.compile(r"(?:https?://)?(?:www\.)?twitter\.com/([A-Za-z0-9_]{1,15})")
# Simple blog/website finder (quotes inside the character class escaped by switching to single-quoted raw string)
_BLOG_URL_RE = re.compile(r'(?:blog|website|homepage)[:\s]*?(https?://[^\s\'"]+)', re.IGNORECASE) # Simple blog/website finder

# Files often containing contact info
_CONTACT_FILES = ["README.md", "README", "CONTRIBUTING.md", "AUTHORS", "AUTHORS.txt", "CODE_OF_CONDUCT.md", "SECURITY.md", "LICENSE", "package.json", "setup.py", "pyproject.toml"]

class ContactScraper:
    """Best‑effort heuristics to pull maintainer contact info."""

    def _safe_decode(self, content: bytes) -> str:
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return content.decode('latin-1') # Try fallback encoding
            except Exception:
                 print("[yellow]Warning: Could not decode file content.")
                 return "" # Give up if both fail

    def _extract_from_text(self, text: str, existing_emails: Set[str]) -> ContactInfo:
        """Extracts contacts, avoiding duplicates and common non-contact emails."""
        emails = set(_EMAIL_RE.findall(text))
        # Filter out potential noise or common non-personal emails
        filtered_emails = {
            email for email in emails
            if not email.endswith(('@users.noreply.github.com', '@example.com', '.png', '.jpg', '.gif'))
               and email not in existing_emails
        }

        tw_match = _TWITTER_RE.search(text)
        twitter = tw_match.group(1) if tw_match else None

        blog_match = _BLOG_URL_RE.search(text)
        blog = blog_match.group(1) if blog_match else None

        return ContactInfo(emails=list(filtered_emails), twitter=twitter, blog=blog)

    def _get_file_content(self, repo: Repository, filepath: str) -> Optional[str]:
        try:
            content_file: ContentFile | list[ContentFile] = repo.get_contents(filepath) # Type hint fix
            if isinstance(content_file, ContentFile) and content_file.content: # Ensure it's a file with content
                 return self._safe_decode(content_file.decoded_content)
            elif isinstance(content_file, list): # It's a directory listing
                 print(f"[yellow]Warning: Expected file, got directory listing for {filepath}")
                 return None
            else:
                 # Handle unexpected types or empty files gracefully
                 print(f"[yellow]Warning: Could not get valid content for {filepath}")
                 return None

        except GithubException as e:
            if e.status == 404:
                # print(f"File not found: {filepath}") # Common, maybe don't log
                pass
            else:
                print(f"[yellow]Warning: GitHub error getting {filepath}: {e.status} {e.data}")
            return None
        except Exception as e:
            print(f"[red]Error processing file {filepath}: {e}")
            return None


    def extract(self, repo: Repository) -> ContactInfo:
        """Extracts contact info from owner profile and common project files."""
        found_emails: Set[str] = set()
        found_twitter: Optional[str] = None
        found_blog: Optional[str] = None

        # 1. Try owner profile
        try:
            owner: NamedUser = repo.owner
            if owner.email:
                # Basic validation for owner email
                if _EMAIL_RE.match(owner.email) and '@users.noreply.github.com' not in owner.email:
                      found_emails.add(owner.email)
            if owner.twitter_username:
                 found_twitter = owner.twitter_username
            if owner.blog:
                # Basic validation for blog URL
                if owner.blog.startswith(('http://', 'https://')):
                     found_blog = owner.blog
        except Exception as e:
            print(f"[yellow]Warning: Could not access owner details for {repo.full_name}: {e}")


        # 2. Scrape common files
        for filename in _CONTACT_FILES:
            content = self._get_file_content(repo, filename)
            if content:
                sub_contact = self._extract_from_text(content, found_emails)
                found_emails.update(sub_contact.emails)
                if not found_twitter and sub_contact.twitter:
                    found_twitter = sub_contact.twitter
                if not found_blog and sub_contact.blog:
                    found_blog = sub_contact.blog # Prioritize owner blog if found

        # 3. (Optional) Scrape recent commit authors - can be noisy/slow
        # try:
        #    commits = repo.get_commits()
        #    for commit in commits[:10]: # Check last 10 commits
        #        if commit.author and commit.author.email and _EMAIL_RE.match(commit.author.email):
        #             if '@users.noreply.github.com' not in commit.author.email:
        #                 found_emails.add(commit.author.email)
        # except Exception as e:
        #    print(f"[yellow]Warning: Could not fetch commit history for {repo.full_name}: {e}")


        final_contact = ContactInfo(
            emails=list(sorted(found_emails)),
            twitter=found_twitter,
            blog=found_blog
        )
        # print(f"Extracted for {repo.full_name}: {final_contact}") # Debug log
        return final_contact

