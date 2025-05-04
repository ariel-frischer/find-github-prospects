import json
import logging
import multiprocessing  # Import for Lock type hint
import re
import traceback
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import litellm
from bs4 import BeautifulSoup
from goose3 import Goose
from pydantic import ValidationError

# Import configuration for model name
from .config import SUMMARIZER_LLM_MODEL

# Import the new Pydantic model and cost tracking helper
from .models import UrlSummarySchema
from .utils import CostDataType, _update_shared_cost  # Import helper and type hint

# --- Configuration ---
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
    url: str,
    content: str,
    issue_title: str,
    issue_body_context: str,
    shared_cost_data: CostDataType,  # Add shared dict
    lock: multiprocessing.Lock,  # Add lock
) -> Optional[UrlSummarySchema]:
    """
    Summarizes scraped content using an LLM, updates shared cost data,
    and expects a JSON response conforming
    to UrlSummarySchema.

    Args:
        url: The URL the content was scraped from.
        content: The scraped text content.
        issue_title: The title of the original GitHub issue.
        issue_body_context: Plain text context from the issue body.

    Returns:
        A validated UrlSummarySchema object if successful, otherwise None.
    """
    logger.info(f"Requesting LLM summary/relevance check for: {url}")
    # Truncate content if necessary
    truncated_content = content[:MAX_CONTENT_LENGTH_FOR_SUMMARY]
    if len(content) > MAX_CONTENT_LENGTH_FOR_SUMMARY:
        truncated_content += " ... (truncated due to length)"

    # Get schema as JSON string
    schema_json = json.dumps(UrlSummarySchema.model_json_schema(), indent=2)

    # Construct prompt asking for JSON output
    summarization_prompt = f"""
Analyze the following 'Content from linked URL' in the context of the 'Original GitHub Issue'.

Original GitHub Issue Title: {issue_title}
Original GitHub Issue Body Context:
{issue_body_context}

---
Content from linked URL ({url}):
{truncated_content}

---
Task: Determine if the 'Content from linked URL' is relevant to understanding or resolving the 'Original GitHub Issue'.
If it IS relevant, provide a concise summary focusing *only* on the relevant aspects.
If it IS NOT relevant, indicate that.

Generate **ONLY** a valid JSON object conforming strictly to the following schema. Do not include any text before or after the JSON object.

Required JSON Output Schema:
```json
{schema_json}
```

JSON Output:
"""
    raw_llm_output: Optional[str] = None
    try:
        # Call LLM, requesting structured output via Pydantic model
        response = litellm.completion(
            model=SUMMARIZER_LLM_MODEL,
            messages=[{"role": "user", "content": summarization_prompt}],
            response_format=UrlSummarySchema,  # Request structured output
        )

        # --- Call Cost Tracking Helper ---
        _update_shared_cost(response, shared_cost_data, lock, f"URL Summary ({url})")
        # --- End Cost Tracking Call ---

        # LiteLLM returns the parsed Pydantic object directly when response_format is used
        if response.choices and response.choices[0].message.content:
            # The content should already be the parsed Pydantic object
            # However, LiteLLM's type hints might be generic, so we validate just in case
            # (or if the API response structure changes)
            llm_data = response.choices[0].message.content
            if isinstance(llm_data, UrlSummarySchema):
                validated_data = llm_data
            elif isinstance(llm_data, dict):  # Fallback if it returns dict
                validated_data = UrlSummarySchema.model_validate(llm_data)
            elif isinstance(llm_data, str):  # Fallback if it returns string
                raw_llm_output = llm_data.strip()
                parsed_json = json.loads(raw_llm_output)
                validated_data = UrlSummarySchema.model_validate(parsed_json)
            else:
                raise TypeError(
                    f"Unexpected response content type from LiteLLM: {type(llm_data)}"
                )

            logger.info(
                f"LLM relevance/summary successful for {url}: Relevant={validated_data.is_relevant}"
            )
            # Log the full response object if relevant
            if validated_data.is_relevant:
                logger.info(f"  Relevant URL Summary Response: {validated_data}")
            return validated_data
        else:
            logger.warning(f"LLM returned empty or invalid response for: {url}")
            return None

    except json.JSONDecodeError as json_e:
        logger.error(
            f"Could not parse LLM JSON output for {url}: {json_e}. Raw response: {raw_llm_output}"
        )
        return None
    except ValidationError as validation_e:
        logger.error(
            f"LLM output failed Pydantic validation for {url}: {validation_e}. Raw response: {raw_llm_output}"
        )
        return None
    except Exception as e:
        logger.error(
            f"LLM summarization failed for URL {url}: {e}\n{traceback.format_exc()}"
        )
        return None


def process_urls_for_issue(
    body_html: str,
    comments_html: List[str],
    issue_title: str,
    base_issue_url: str,
    shared_cost_data: CostDataType,  # Add shared dict
    lock: multiprocessing.Lock,  # Add lock
) -> List[Dict[str, str]]:
    """
    Extracts relevant URLs, scrapes them, summarizes relevant content (updating shared cost data),
    and returns a list of *relevant* summaries.
    summarizes relevant content, and returns a list of *relevant* summaries.

    Args:
        body_html: The HTML content of the issue body.
        comments_html: A list of HTML strings, one for each comment.
        issue_title: The title of the original issue for context.
        base_issue_url: The URL of the issue page, used for resolving relative links.

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
            return url_summaries  # Return empty list if no URLs found

        # Attempt to get plain text from body HTML for better context
        try:
            soup = BeautifulSoup(body_html, "html.parser")
            issue_body_context = soup.get_text(separator=" ", strip=True)
            if not issue_body_context:
                issue_body_context = (
                    "Could not extract plain text from issue body HTML."
                )
            logger.debug("Extracted plain text context from body HTML for summarizer.")
        except Exception:
            issue_body_context = "Error extracting plain text from issue body HTML."
            logger.warning(f"{issue_body_context}", exc_info=True)

        for url in unique_urls:
            content = _scrape_url_content(url)
            if content:
                # Call the updated summarizer function, passing shared objects
                summary_result: Optional[UrlSummarySchema] = _summarize_content(
                    url,
                    content,
                    issue_title,
                    issue_body_context,
                    shared_cost_data,
                    lock,
                )

                # Check the result and add to list only if relevant and summary exists
                if (
                    summary_result
                    and summary_result.is_relevant
                    and summary_result.summary
                ):
                    url_summaries.append(
                        {"url": url, "summary": summary_result.summary}
                    )
                    # Log the actual summary content
                    logger.info(f"Added relevant summary for URL: {url}")
                    logger.info(
                        f"  Summary: {summary_result.summary[:100]}..."
                    )  # Log first 100 chars
                elif summary_result and not summary_result.is_relevant:
                    logger.info(f"URL content deemed irrelevant by LLM: {url}")
                else:
                    logger.warning(f"No valid/relevant summary obtained for URL: {url}")

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
