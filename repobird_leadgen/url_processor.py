import logging
import os
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import litellm
from bs4 import BeautifulSoup
from goose3 import Goose

# Removed URLExtract import as we now parse HTML
# from urlextract import URLExtract

# --- Configuration ---
# LLM specifically for URL summarization (Enricher uses ENRICHER_LLM_MODEL from config)
SUMMARIZER_LLM_MODEL = os.getenv(
    "SUMMARIZER_LLM_MODEL", "openrouter/deepseek/deepseek-chat-v3-0324"
)
MAX_CONTENT_LENGTH_FOR_SUMMARY = 200000  # Limit input to summarizer LLM

# Get logger for this module
logger = logging.getLogger(__name__)


# --- URL Extraction from HTML ---
def _is_relevant_github_link(url: str, base_repo_url: str) -> bool:
    """Check if a GitHub URL is likely relevant (issue, PR, external repo) and not just navigation."""
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()

    # Ignore common non-content links
    ignore_paths = {
        "/login",
        "/signup",
        "/notifications",
        "/settings",
        "/explore",
        "/marketplace",
        "/new",
        "/codespaces",
        "/pulls",
        "/issues",  # Ignore top-level /issues and /pulls
        "/projects",
        "/actions",
        "/security",
        "/pulse",
        "/graphs",
        "/find",
        "/topics",
    }
    if path in ignore_paths:
        return False
    if path.startswith(("/settings/", "/codespaces/", "/topics/")):
        return False

    # Ignore links within the same repo that aren't issues or PRs (e.g., code files, commits)
    # Assumes base_repo_url is like 'https://github.com/owner/repo'
    parsed_base = urlparse(base_repo_url)
    base_path = parsed_base.path.lower()
    if path.startswith(base_path):
        sub_path = path[len(base_path) :].strip("/")
        # Allow specific subpaths like issues/NUMBER, pulls/NUMBER, releases/tag/TAG
        if not (
            re.match(r"^issues/\d+$", sub_path)
            or re.match(r"^pull/\d+$", sub_path)
            or re.match(r"^releases/tag/", sub_path)
            or re.match(r"^discussions/\d+$", sub_path)
            or sub_path
            == ""  # Link to the repo root itself (might be relevant context)
        ):
            # Check for common code/commit paths
            if sub_path.startswith(("blob/", "tree/", "commit/", "commits/")):
                return False
            # Check for label/milestone/assignee links
            if sub_path.startswith(("labels/", "milestones/", "assignees/")):
                return False

    # Ignore user profile links (heuristic: path is just /username)
    if re.match(r"^/[a-zA-Z0-9-]+$", path) and not re.search(
        r"/(issues|pull|releases|discussions)/", path
    ):
        # More robust check might involve checking if it's a known org/user via API if needed
        return False

    return True


def _extract_urls_from_html(html_content: str, base_url: str) -> List[str]:
    """
    Extracts and resolves relevant URLs from an HTML string using BeautifulSoup.
    Filters out common GitHub navigation, fragments, and less relevant links.

    Args:
        html_content: The HTML string to parse.
        base_url: The base URL (e.g., the issue page URL) to resolve relative links.

    Returns:
        A list of unique, absolute, relevant URLs found in the HTML.
    """
    urls = set()
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        base_repo_url = "/".join(
            base_url.split("/")[:5]
        )  # Heuristic: https://github.com/owner/repo

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # 1. Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            # 2. Basic cleanup and validation
            parsed_url = urlparse(absolute_url)
            if parsed_url.scheme not in ["http", "https"]:
                continue  # Ignore non-http(s) schemes (mailto, javascript, etc.)
            if parsed_url.fragment:
                # Optional: Keep fragment if needed, but usually not for scraping content
                # For now, remove fragment for cleaner URLs
                absolute_url = absolute_url.split("#")[0]

            # 3. Filter out irrelevant links
            is_github = "github.com" in parsed_url.netloc.lower()
            if is_github:
                if not _is_relevant_github_link(absolute_url, base_repo_url):
                    logger.debug(f"Filtering out GitHub link: {absolute_url}")
                    continue
            # Add more general filters if needed (e.g., specific domains)

            urls.add(absolute_url)

    except Exception as e:
        logger.warning(f"Error parsing HTML for URL extraction: {e}", exc_info=True)

    logger.debug(f"Extracted URLs from HTML: {list(urls)}")
    return list(urls)


def _scrape_url_content(url: str) -> Optional[str]:
    """Scrapes the main text content from a URL using Goose."""
    try:
        logger.info(f"Scraping URL content: {url}")
        g = Goose()
        article = g.extract(url=url)
        main_text = article.cleaned_text
        if main_text:
            logger.info(f"Scraped content from: {url}")
            return main_text
        else:
            logger.info(f"No significant text found at: {url}")
            return None
    except Exception as e:
        logger.warning(f"Failed to scrape URL {url}: {e}", exc_info=False)
        return None


def _summarize_content(
    url: str, content: str, issue_title: str, issue_body: str
) -> Optional[str]:
    """Summarizes the scraped content using an LLM, focusing on relevance to the issue."""
    try:
        logger.info(f"Summarizing content from: {url}")
        # Truncate content if necessary and add indicator
        truncated_content = content[:MAX_CONTENT_LENGTH_FOR_SUMMARY]
        if len(content) > MAX_CONTENT_LENGTH_FOR_SUMMARY:
            truncated_content += " ... (truncated due to length)"

        # Construct summarization prompt
        summarization_prompt = f"""
        Original GitHub Issue Title: {issue_title}
        Original GitHub Issue Body:
        {issue_body}

        ---
        Content from linked URL ({url}):
        {truncated_content}

        ---
        Task: Summarize the above 'Content from linked URL', focusing *only* on aspects relevant to the 'Original GitHub Issue Title' and 'Original GitHub Issue Body'. If the content is irrelevant, state that clearly (e.g., "Content is irrelevant to the issue.").
        """
        # Call LLM for summarization
        response = litellm.completion(
            model=SUMMARIZER_LLM_MODEL,  # Use summarizer-specific model
            messages=[{"role": "user", "content": summarization_prompt}],
            # max_tokens=150,  # Keep summaries concise
            # temperature=0.2,  # Low temperature for factual summary
        )
        # response_cost = getattr(response, "response_cost", None) # Removed print
        # print(f"response_cost={response_cost}") # Removed print
        summary = response.choices[0].message.content.strip()
        if summary:
            logger.info(f"Summarized: {url}")
            logger.info(f"  Summary for {url}: {summary}")  # Log the summary content
            return summary
        else:
            logger.warning(f"LLM returned empty summary for: {url}")
            return None
    except Exception as e:
        logger.warning(f"LLM summarization failed for URL {url}: {e}", exc_info=True)
        return None


def process_urls_for_issue(
    body_html: str,
    comments_html: List[str],
    issue_title: str,
    base_issue_url: str,
) -> List[Dict[str, str]]:
    """
    Extracts relevant URLs from issue body and comment HTML, scrapes them,
    summarizes relevant content, and returns a list of summaries.

    Args:
        body_html: The HTML content of the issue body.
        comments_html: A list of HTML strings, one for each comment.
        issue_title: The title of the original issue for context.
        base_issue_url: The URL of the issue page, used for resolving relative links
                        and logging.

    Returns:
        A list of dictionaries, where each dictionary contains 'url' and 'summary'.
    """
    url_summaries: List[Dict[str, str]] = []
    all_urls = set()

    try:
        # Extract URLs from body HTML
        body_urls = _extract_urls_from_html(body_html, base_issue_url)
        all_urls.update(body_urls)

        # Extract URLs from comments HTML
        for comment_html in comments_html:
            comment_urls = _extract_urls_from_html(comment_html, base_issue_url)
            all_urls.update(comment_urls)

        unique_urls = list(all_urls)
        logger.info(
            f"Found {len(unique_urls)} unique, relevant URLs in issue: {base_issue_url}"
        )

        if not unique_urls:
            return url_summaries

        # Use issue title and *plain text* of body for summarization context
        # (We don't have plain text body easily here, maybe pass it or scrape it again lightly?)
        # For now, let's just use the title as primary context for the summarizer.
        # A better approach might involve getting plain text from the HTML body earlier.
        issue_body_context = "Body context not available for summarizer."  # Placeholder

        for url in unique_urls:
            # Removed the explicit skip for github.com URLs to allow scraping linked issues/PRs.
            # if "github.com" in urlparse(url).netloc.lower():
            #     logger.info(f"Skipping scraping of internal GitHub URL: {url}")
            #     continue

            content = _scrape_url_content(url)
            if content:
                # Pass title and placeholder body context to summarizer
                summary = _summarize_content(
                    url, content, issue_title, issue_body_context
                )
                if summary:
                    url_summaries.append({"url": url, "summary": summary})
            # Consider adding delay here if needed: time.sleep(0.5)

    except Exception as e:
        logger.error(
            f"Error during URL processing for issue {base_issue_url}: {e}",
            exc_info=True,
        )
    finally:
        logger.info(
            f"Finished URL processing for {base_issue_url}. Found {len(url_summaries)} relevant summaries (including internal GitHub links)."
        )
        return url_summaries
