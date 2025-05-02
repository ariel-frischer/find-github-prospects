import json
import logging
import traceback
from dataclasses import (
    dataclass,
    field,
)
from typing import Any, Dict, List, Optional  # Ensured present

import litellm
from playwright.sync_api import (
    Error as PlaywrightError,  # Import PlaywrightError
)
from playwright.sync_api import (
    Page,
    Playwright,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,  # Import PlaywrightTimeoutError
)
from pydantic import ValidationError

# Import configuration
from .config import ENRICHER_LLM_MODEL

# Removed URLExtract and Goose imports as they are now in url_processor
# from urlextract import URLExtract
# from goose3 import Goose
# Import the Pydantic models from models.py
from .models import IssueAnalysis, LLMIssueAnalysisSchema

# Import the new URL processing function
from .url_processor import process_urls_for_issue

# --- Configuration ---
# Ensure API keys (e.g., OPENROUTER_API_KEY, GOOGLE_API_KEY) are set in your environment for LiteLLM
# LLM model is now configured via config.py (ENRICHER_LLM_MODEL)
MAX_COMMENTS_FOR_LLM = 100  # Limit number of comments sent to LLM
PLAYWRIGHT_TIMEOUT = 30000  # 30 seconds for page operations

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Data Structures ---
# IssueAnalysis is now imported from models.py


@dataclass
class ScrapedIssueData:
    """Raw data scraped from a GitHub issue page."""

    title: Optional[str] = None
    body_html: Optional[str] = None  # Store body as HTML
    comments_html: List[str] = field(default_factory=list)  # Store comments as HTML
    error: Optional[str] = None


# --- Enricher Class ---
class Enricher:
    """
    Scrapes GitHub issues identified in a cache file, analyzes them using an LLM,
    and returns structured analysis results.
    """

    def __init__(self, headless: bool = True, timeout: int = PLAYWRIGHT_TIMEOUT):
        self.headless = headless
        self.timeout = timeout
        self.playwright: Optional[Playwright] = None
        self.browser = None
        # Configure LiteLLM (ensure API keys are set in environment)
        # litellm.set_verbose=True # Optional: for debugging LiteLLM calls

    def __enter__(self):
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            logging.info("Playwright browser started.")
        except Exception as e:
            logging.error(f"Error starting Playwright: {e}")
            self._cleanup()  # Ensure cleanup if launch fails
            raise  # Re-raise the exception
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        logging.info("Playwright browser stopped.")

    def _cleanup(self):
        """Safely closes browser and stops Playwright."""
        if self.browser:
            try:
                self.browser.close()
                self.browser = None
            except Exception as e:
                logging.warning(f"Warning: Error closing Playwright browser: {e}")
        if self.playwright:
            try:
                self.playwright.stop()
                self.playwright = None
            except Exception as e:
                logging.warning(f"Warning: Error stopping Playwright: {e}")

    def _get_new_page(self) -> Page:
        """Creates a new browser page."""
        if not self.browser:
            raise RuntimeError(
                "Browser not initialized. Ensure Enricher is used within a context manager."
            )
        page = self.browser.new_page()
        page.set_default_timeout(self.timeout)
        return page

    def scrape_github_issue(self, issue_url: str) -> ScrapedIssueData:
        """Scrapes the title, body, and comments from a GitHub issue URL."""
        page = None
        try:
            page = self._get_new_page()
            logging.info(f"  Scraping issue: {issue_url}")
            page.goto(issue_url, wait_until="domcontentloaded", timeout=self.timeout)
            logging.info("    Page loaded. Waiting for container elements...")

            # Define selectors based on browser_checker.py and stable attributes
            title_container_selector = 'div[data-component="TitleArea"]'
            # Use data-testid for the body container
            body_container_selector = 'div[data-testid="issue-body"]'
            # Specific title and body elements within containers
            # Target the BDI element containing the title text using its data-testid
            title_selector = 'bdi[data-testid="issue-title"]'
            body_selector = (
                'div.markdown-body[data-testid="markdown-body"]'  # Body content area
            )
            # Updated comment selectors based on observed HTML structure
            # Select the outer container for each timeline item (comment, event, etc.)
            comment_container_selector = (
                "div.LayoutHelpers-module__timelineElement--tFGhF"
            )
            # Select the markdown body within a comment item
            comment_body_selector = 'div[data-testid="markdown-body"]'

            # 1. Wait for container elements to be attached (ensures structure exists)
            logging.info(
                f"    Waiting for title container: '{title_container_selector}'"
            )
            page.wait_for_selector(
                title_container_selector, state="attached", timeout=self.timeout
            )
            logging.info(f"    Waiting for body container: '{body_container_selector}'")
            # Wait for the *first* comment container, which usually holds the body
            page.locator(body_container_selector).first.wait_for(
                state="attached", timeout=self.timeout
            )
            logging.info("    Container elements attached.")

            # 2. Wait for specific title and body elements to be visible
            logging.info(
                f"    Waiting for title element to be visible: '{title_selector}'"
            )
            page.wait_for_selector(
                title_selector, state="visible", timeout=self.timeout
            )
            title_elem = page.locator(title_selector).first
            title = title_elem.text_content(timeout=self.timeout / 2) or ""
            title = title.strip()
            logging.info(f"    Title found: '{title[:50]}...'")

            logging.info(
                f"    Waiting for body element to be visible: '{body_selector}'"
            )
            # Target the body within the *first* comment container
            body_elem = (
                page.locator(body_container_selector).first.locator(body_selector).first
            )
            body_elem.wait_for(state="visible", timeout=self.timeout)
            # Get inner HTML for body instead of text_content
            body_html = body_elem.inner_html(timeout=self.timeout / 2) or ""
            body_html = body_html.strip()
            logging.info(f"    Body HTML found (length: {len(body_html)}).")

            # 3. Get comments (using updated selectors and logic)
            logging.info(
                f"    Locating comment containers: '{comment_container_selector}'"
            )
            comments_html = []  # Store comment HTML
            # Locate all timeline elements
            timeline_elements = page.locator(comment_container_selector)
            timeline_count = timeline_elements.count()
            logging.info(f"    Found {timeline_count} potential timeline elements.")

            # REMOVED: Unused variable assignment for main_body_element
            # main_body_element = page.locator(body_container_selector).first

            for i in range(timeline_count):
                timeline_element = timeline_elements.nth(i)

                # Check if this timeline element contains the main issue body
                potential_body_in_timeline = timeline_element.locator(
                    body_selector
                ).first
                is_main_body = False
                if potential_body_in_timeline.count() > 0:
                    current_body_elem = timeline_element.locator(
                        comment_body_selector
                    ).first
                    # Check if the HTML content matches the scraped body HTML
                    if current_body_elem.count() > 0:
                        current_body_html = (
                            current_body_elem.inner_html(timeout=self.timeout / 4) or ""
                        )
                        if (
                            current_body_html.strip() == body_html
                        ):  # Compare with the main body HTML
                            if not comments_html:  # If we haven't added any comments yet, this is likely the main body
                                logging.info(
                                    f"    Skipping timeline element {i} (identified as main issue body)."
                                )
                                is_main_body = True

                if not is_main_body:
                    # This is likely a comment, try to extract its body
                    # Get inner HTML for comments
                    comment_body_elem = timeline_element.locator(
                        comment_body_selector
                    ).first
                    if comment_body_elem.count() > 0:
                        html_content = (
                            comment_body_elem.inner_html(timeout=self.timeout / 2) or ""
                        )
                        html_content = html_content.strip()
                        if html_content:
                            logging.info(
                                f"    Found comment HTML in timeline element {i} (length: {len(html_content)})."
                            )
                            comments_html.append(html_content)

            logging.info(
                f"    Scraped: Title='{title[:50]}...', Body HTML length={len(body_html)}, Comments HTML count={len(comments_html)}"
            )
            return ScrapedIssueData(
                title=title, body_html=body_html, comments_html=comments_html
            )

        except Exception as e:
            error_msg = f"Error scraping {issue_url}: {e}\n{traceback.format_exc()}"
            logging.warning(f"    Warning: {error_msg}")
            return ScrapedIssueData(error=error_msg)
        finally:
            if page:
                try:
                    page.close()
                except Exception as e:
                    logging.warning(f"Warning: Error closing page for {issue_url}: {e}")

    def scrape_readme_text(self, repo_url: str) -> Optional[str]:
        """
        Navigates to the repository URL and scrapes the text content of the README.
        (Similar to the method in BrowserIssueChecker)

        Args:
            repo_url: The full URL to the repository's main page.

        Returns:
            The text content of the README, or None if not found or an error occurs.
        """
        if not self.browser or not self.playwright:
            logging.error(
                "[Enricher] Playwright/Browser not initialized. Cannot scrape README."
            )
            return None

        page = None
        readme_selector = "article.markdown-body"  # Selector for the README content

        try:
            logging.info(f"  Scraping README: {repo_url}")
            page = self._get_new_page()
            page.goto(repo_url, wait_until="domcontentloaded", timeout=self.timeout)
            logging.info("    README page loaded. Waiting for selector...")

            # Wait for the README container to be visible
            readme_element = page.locator(readme_selector).first
            readme_element.wait_for(state="visible", timeout=self.timeout)
            logging.info("    README element located.")

            # Extract text content
            readme_text = readme_element.text_content(timeout=self.timeout / 2)
            logging.info(
                f"    README text extracted (length: {len(readme_text or '')})."
            )
            return readme_text.strip() if readme_text else None

        except PlaywrightTimeoutError:
            logging.warning(f"    Timeout waiting for README on {repo_url}")
            return None
        except PlaywrightError as pe:
            logging.warning(
                f"    Playwright error scraping README from {repo_url}: {pe}"
            )
            return None
        except Exception as e:
            logging.error(
                f"    Unexpected error scraping README from {repo_url}: {e}",
                exc_info=True,
            )
            return None
        finally:
            if page:
                try:
                    page.close()
                except Exception as e:
                    logging.warning(
                        f"    Warning: Error closing page after README scrape: {e}"
                    )

    def _build_prompt(
        self,
        title: Optional[str],  # Allow None
        body_html: Optional[str],  # Changed from body
        comments_html: List[str],  # Changed from comments
        url_summaries: List[Dict[str, str]],
        readme_text: Optional[str],  # Add readme_text parameter
    ) -> str:
        """Builds the prompt for the LLM analysis, including URL summaries and README."""
        if not title:  # Keep this check
            return ""  # Cannot analyze without a title

        # Limit comments sent to LLM (using HTML comments now)
        # Note: Sending raw HTML in comments might exceed token limits faster.
        # Consider extracting text from HTML here if needed, or truncating more aggressively.
        comments_str = " || ".join(comments_html[:MAX_COMMENTS_FOR_LLM])
        if len(comments_html) > MAX_COMMENTS_FOR_LLM:
            comments_str += f" || ... (truncated, {len(comments_html) - MAX_COMMENTS_FOR_LLM} more comments)"

        # Format URL Summaries
        url_summaries_text = "**Summaries of Linked URLs:**\n"
        if url_summaries:
            for item in url_summaries:
                url_summaries_text += (
                    f"- URL: {item['url']}\n  Summary: {item['summary']}\n"
                )
        else:
            url_summaries_text += "No relevant external content was found or summarized from linked URLs.\n"
        url_summaries_text += "\n---\n"  # Separator

        # Generate the schema dynamically based on whether README exists
        # Start with the base model schema
        schema_dict = LLMIssueAnalysisSchema.model_json_schema()
        # If no readme_text, remove the optional readme_summary field from the schema shown to the LLM
        if not readme_text:
            if "readme_summary" in schema_dict.get("properties", {}):
                del schema_dict["properties"]["readme_summary"]
            # Also remove from required list if it somehow ended up there (it shouldn't be)
            if "readme_summary" in schema_dict.get("required", []):
                schema_dict["required"].remove("readme_summary")

        schema_json = json.dumps(schema_dict, indent=2)

        # Format README text
        readme_section = "<readme>\n"
        if readme_text:
            # Optional: Truncate README if it's excessively long?
            # max_readme_len = 10000
            # if len(readme_text) > max_readme_len:
            #     readme_text = readme_text[:max_readme_len] + "... (truncated)"
            readme_section += readme_text
        else:
            readme_section += "README content could not be scraped or was empty."
        readme_section += "\n</readme>\n\n---\n"  # Separator

        # Add conditional instruction for readme_summary
        readme_summary_instruction = ""
        if readme_text:
            readme_summary_instruction = "\n- Include a concise `readme_summary` based *only* on the provided README content, focusing on project purpose and context relevant to the issue."
        else:
            readme_summary_instruction = "\n- **Do not** include the `readme_summary` field in your JSON output, as no README content was provided."

        prompt = f"""
Analyze the following GitHub issue content (title, body, comments), summaries of linked URLs, and the repository README (if provided) to determine the issue's suitability as a task potentially solvable by an autonomous AI software agent. Focus on understanding the core problem, its complexity, requirements, and the overall context provided by the README.

**Repository README Content:**

{readme_section}

**Input Issue Content:**

*   **Title:** {title}
*   **Body (HTML):**
    ```html
    {body_html}
    ```
*   **Comments (HTML, selected):**
    ```html
    {comments_str}
    ```

---

{url_summaries_text}

**Instructions:**

Your task is to analyze the provided GitHub issue content (title, body, comments), the summaries of linked URLs, AND the repository README content (if provided).
Based on your combined analysis, generate **ONLY** a valid JSON object that strictly conforms to the following JSON schema.
{readme_summary_instruction}
**DO NOT** include any text, explanations, apologies, summaries, or markdown formatting before or after the JSON object. Your entire response must be the JSON object itself.

**Required JSON Output Schema:**

```json
{schema_json}
```

**Generate the JSON output now:**
"""
        return prompt

    def _analyze_issue_with_llm(
        self,
        scraped_data: ScrapedIssueData,
        url_summaries: List[Dict[str, str]],
        readme_text: Optional[str],  # Add readme_text parameter
    ) -> Dict[str, Any]:
        """
        Analyzes the scraped issue data, URL summaries, and README using LiteLLM, expecting a JSON object
        that conforms to the LLMIssueAnalysisSchema Pydantic model.

        Returns:
            A dictionary containing the validated analysis data or an 'error' key.
        """
        prompt = self._build_prompt(
            scraped_data.title,
            scraped_data.body_html,  # Use body_html
            scraped_data.comments_html,  # Use comments_html
            url_summaries,
            readme_text,  # Pass the readme_text argument here
        )
        if not prompt:
            return {"error": "Cannot analyze issue without a title."}

        logging.info(f"    Analyzing with LLM ({ENRICHER_LLM_MODEL})...")
        analysis_result: Optional[LLMIssueAnalysisSchema] = None
        error_message: Optional[str] = None
        raw_llm_output: Optional[str] = None

        try:
            response = litellm.completion(
                model=ENRICHER_LLM_MODEL,  # Use configured enricher model
                messages=[{"role": "user", "content": prompt}],
                response_format=LLMIssueAnalysisSchema,
            )
            logging.info(f"response={response}")
            # Extract the model's JSON output string
            raw_llm_output = response.choices[0].message.content

            # Parse the raw JSON string and validate against the Pydantic model
            json_string = raw_llm_output.strip()
            llm_data = json.loads(json_string)
            analysis_result = LLMIssueAnalysisSchema.model_validate(llm_data)
            logging.info("    LLM Analysis successful and validated.")
            # Return the validated data as a dictionary
            return analysis_result.model_dump()

        except json.JSONDecodeError as json_e:
            error_message = f"Could not parse LLM JSON output: {json_e}. Raw response: {raw_llm_output}"
        except ValidationError as validation_e:
            error_message = f"LLM output failed Pydantic validation: {validation_e}. Raw response: {raw_llm_output}"
        except Exception as e:
            # Catch other potential LiteLLM errors (API keys, network issues, etc.)
            error_message = (
                f"Error during LiteLLM completion: {e}\n{traceback.format_exc()}"
            )

        # Handle errors
        if error_message:
            logging.error(f"    Error: {error_message}")
            return {"error": error_message}
        else:
            # Should not happen if logic is correct, but as a fallback
            return {"error": "Unknown error during LLM analysis."}

    def process_repo(self, repo_data: Dict[str, Any]) -> List[IssueAnalysis]:
        """
        Processes a single repository's data: scrapes and analyzes its found issues.

        Args:
            repo_data: A dictionary representing a repository's data from the cache file,
                       expected to contain 'full_name', 'html_url', and 'found_issues'.

        Returns:
            A list of IssueAnalysis objects for the processed issues.
        """
        repo_name = repo_data.get("full_name", "Unknown Repo")
        repo_html_url = repo_data.get("html_url")
        found_issues = repo_data.get("found_issues")
        analysis_results: List[IssueAnalysis] = []

        if not repo_html_url or not isinstance(found_issues, list) or not found_issues:
            logging.warning(
                f"Skipping enrichment for {repo_name}: Missing 'html_url' or valid 'found_issues' list."
            )
            return analysis_results  # Return empty list

        logging.info(f"Processing repo: {repo_name}")

        # --- Scrape README first ---
        readme_text = self.scrape_readme_text(repo_html_url)
        # --- End README scrape ---

        logging.info(f"Processing issues for repo: {repo_name}")
        # Iterate through the list of issue numbers directly
        for issue_number in found_issues:
            # Construct the issue URL
            issue_url = f"{repo_html_url}/issues/{issue_number}"
            logging.debug(
                f"  Processing issue number: {issue_number}, URL: {issue_url}"
            )

            # Basic validation of the issue number from the cache
            if not isinstance(issue_number, int) or issue_number <= 0:
                logging.warning(
                    f"  Skipping invalid issue number in {repo_name}: {issue_number}"
                )
                error_str = f"Invalid issue number in cache: {issue_number}"
                analysis = IssueAnalysis(
                    issue_number=issue_number,
                    issue_url=issue_url,  # Still log the constructed URL if possible
                    error=error_str,
                    full_problem_statement="Error: Invalid issue number format in cache.",
                    complexity_score=-1.0,
                    is_good_first_issue_for_agent=False,
                    reasoning="N/A - Invalid cache data",
                    required_domain_knowledge=-1.0,
                    codebase_familiarity_needed=-1.0,
                    scope_clarity=-1.0,
                    meta_notes="Invalid issue data provided in cache.",
                )
                analysis_results.append(analysis)
                continue

            # --- Issue processing starts here ---
            analysis: Optional[IssueAnalysis] = None
            error_str: Optional[str] = None
            url_summaries: List[Dict[str, str]] = []  # Initialize for each issue

            # 1. Scrape Issue Content using the constructed issue_url
            scraped_data = self.scrape_github_issue(issue_url)
            if scraped_data.error:
                error_str = f"Scraping failed: {scraped_data.error}"
                analysis = IssueAnalysis(
                    issue_number=issue_number,
                    issue_url=issue_url,
                    error=error_str,
                    full_problem_statement=f"Error: Failed to scrape issue content. Details: {error_str}",
                    complexity_score=-1.0,
                    is_good_first_issue_for_agent=False,
                    reasoning="N/A - Scraping failure",
                    required_domain_knowledge=-1.0,
                    codebase_familiarity_needed=-1.0,
                    scope_clarity=-1.0,
                    meta_notes="Scraping failure prevented analysis.",
                )
                analysis_results.append(analysis)
                continue

            # --- Start: URL Processing (Using HTML) ---
            url_summaries = []  # Initialize empty list
            if scraped_data.title and (
                scraped_data.body_html or scraped_data.comments_html
            ):
                # Pass HTML content and base URL to the processing function
                url_summaries = process_urls_for_issue(
                    body_html=scraped_data.body_html or "",
                    comments_html=scraped_data.comments_html,
                    issue_title=scraped_data.title,
                    base_issue_url=issue_url,  # Pass the issue URL for resolving relative links
                )
            # --- End: URL Processing ---

            # 2. Analyze Issue Content (including URL Summaries and README)
            llm_result_dict = self._analyze_issue_with_llm(
                scraped_data,
                url_summaries,
                readme_text,  # Pass readme_text
            )

            # 3. Validate and Create Pydantic Object
            if llm_result_dict.get("error"):
                error_str = f"LLM analysis failed: {llm_result_dict['error']}"
                analysis = IssueAnalysis(
                    issue_number=issue_number,
                    issue_url=issue_url,
                    error=error_str,
                    full_problem_statement=f"Error: LLM analysis failed. Details: {error_str}",
                    complexity_score=-1.0,
                    is_good_first_issue_for_agent=False,
                    reasoning="N/A - LLM analysis failure",
                    required_domain_knowledge=-1.0,
                    codebase_familiarity_needed=-1.0,
                    scope_clarity=-1.0,
                    meta_notes="LLM analysis failure prevented structured output.",
                )
            else:
                # Combine LLM results with issue identifiers and validate
                try:
                    analysis_data = {
                        "issue_number": issue_number,
                        "issue_url": issue_url,
                        **llm_result_dict,  # Spread the validated LLM fields
                    }
                    # Validate the combined data against the full IssueAnalysis model
                    analysis = IssueAnalysis.model_validate(analysis_data)
                except ValidationError as val_err:
                    error_str = (
                        f"Internal validation failed after LLM analysis: {val_err}"
                    )
                    logging.error(f"    Error: {error_str}")
                    analysis = IssueAnalysis(
                        issue_number=issue_number,
                        issue_url=issue_url,
                        error=error_str,
                        full_problem_statement=f"Error: Post-LLM validation failed. Details: {error_str}",
                        complexity_score=-1.0,
                        is_good_first_issue_for_agent=False,
                        reasoning="N/A - Post-LLM validation failure",
                        required_domain_knowledge=-1.0,
                        codebase_familiarity_needed=-1.0,
                        scope_clarity=-1.0,
                        meta_notes="Post-LLM validation failure prevented saving analysis.",
                    )

            if analysis:  # Ensure analysis object was created
                analysis_results.append(analysis)

        logging.info(
            f"Finished processing {len(found_issues)} issue(s) for {repo_name}."
        )
        return analysis_results


# --- Helper function for parallel processing ---


def enrich_repo_entry(repo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function to instantiate Enricher and process a single repo's data.
    Designed for use with parallel_map. Handles Enricher context management.

    Args:
        repo_data: A dictionary representing a repository's data from the cache file.

    Returns:
        The original repo_data dictionary updated with an 'issue_analysis' key
        containing the list of IssueAnalysis results (as dicts).
    """
    enriched_data = repo_data.copy()  # Start with original data
    analysis_list = []
    try:
        with Enricher() as enricher:
            # process_repo now returns a list of Pydantic IssueAnalysis objects
            analysis_results: List[IssueAnalysis] = enricher.process_repo(repo_data)
            # Convert Pydantic objects to dictionaries for JSON serialization using model_dump
            analysis_list = [
                analysis.model_dump(mode="json") for analysis in analysis_results
            ]
    except Exception as e:
        repo_name = repo_data.get("full_name", "Unknown Repo")
        logging.error(
            f"Error processing repo {repo_name} in parallel worker: {e}", exc_info=True
        )
        # Add an error marker to the output data for this repo
        enriched_data["enrichment_error"] = str(e)

    enriched_data["issue_analysis"] = analysis_list
    return enriched_data
