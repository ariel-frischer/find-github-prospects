import logging
import os
from typing import Dict, List, Optional

import litellm
from goose3 import Goose
from urlextract import URLExtract

# --- Configuration ---
# Use the same model as the main enricher for consistency, or define separately if needed
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-1.5-pro-preview-0409")
MAX_CONTENT_LENGTH_FOR_SUMMARY = 200000  # Limit input to summarizer LLM

# Get logger for this module
logger = logging.getLogger(__name__)


def _extract_urls(text: str) -> List[str]:
    """Extracts URLs from a given block of text."""
    try:
        extractor = URLExtract()
        urls = extractor.find_urls(text)
        # Optional: Add filtering here (e.g., ignore github.com URLs, common image hosts?)
        return list(set(urls))  # Return unique URLs
    except Exception as e:
        logger.warning(f"URL extraction failed: {e}", exc_info=True)
        return []


def _scrape_url_content(url: str) -> Optional[str]:
    """Scrapes the main text content from a URL using Goose."""
    try:
        logger.info(f"Attempting to scrape URL: {url}")
        g = Goose()
        article = g.extract(url=url)
        main_text = article.cleaned_text
        if main_text:
            logger.info(f"Successfully scraped content from URL: {url}")
            return main_text
        else:
            logger.info(f"No significant text content found at URL: {url}")
            return None
    except Exception as e:
        logger.warning(f"Failed to scrape URL {url}: {e}", exc_info=False)
        return None


def _summarize_content(
    url: str, content: str, issue_title: str, issue_body: str
) -> Optional[str]:
    """Summarizes the scraped content using an LLM, focusing on relevance to the issue."""
    try:
        logger.info(f"Attempting to summarize content from URL: {url}")
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
            model=LLM_MODEL,  # Use configured model
            messages=[{"role": "user", "content": summarization_prompt}],
            # max_tokens=150,  # Keep summaries concise
            # temperature=0.2,  # Low temperature for factual summary
        )
        # response_cost = getattr(response, "response_cost", None) # Removed print
        # print(f"response_cost={response_cost}") # Removed print
        summary = response.choices[0].message.content.strip()
        if summary:
            logger.info(f"Successfully summarized URL: {url}")
            return summary
        else:
            logger.warning(f"LLM returned empty summary for URL: {url}")
            return None
    except Exception as e:
        logger.warning(
            f"LLM summarization failed for URL {url}: {e}", exc_info=True
        )
        return None


def process_urls_for_issue(
    combined_text: str, issue_title: str, issue_body: str, issue_url_for_logging: str
) -> List[Dict[str, str]]:
    """
    Extracts URLs from text, scrapes them, summarizes relevant content,
    and returns a list of summaries.

    Args:
        combined_text: The text to extract URLs from (e.g., issue title + body + comments).
        issue_title: The title of the original issue for context.
        issue_body: The body of the original issue for context.
        issue_url_for_logging: The URL of the issue, used for logging messages.

    Returns:
        A list of dictionaries, where each dictionary contains 'url' and 'summary'.
    """
    url_summaries: List[Dict[str, str]] = []
    try:
        urls = _extract_urls(combined_text)
        logger.info(
            f"Found {len(urls)} unique URLs in issue {issue_url_for_logging}"
        )

        if not urls:
            return url_summaries

        for url in urls:
            # Optional: Add more robust URL filtering here if needed
            if "github.com" in url:  # Example: Skip GitHub links for now
                logger.info(f"Skipping GitHub URL: {url}")
                continue

            content = _scrape_url_content(url)
            if content:
                summary = _summarize_content(url, content, issue_title, issue_body)
                if summary:
                    url_summaries.append({"url": url, "summary": summary})
            # Add a small delay between URL processing to be polite
            # time.sleep(0.5) # Consider adding if hitting rate limits

    except Exception as e:
        logger.error(
            f"Error during URL processing for issue {issue_url_for_logging}: {e}",
            exc_info=True,
        )
    finally:
        logger.info(
            f"Finished URL processing for {issue_url_for_logging}. Found {len(url_summaries)} relevant summaries."
        )
        return url_summaries
