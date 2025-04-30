import json
import os
import traceback
from dataclasses import (  # Keep asdict for EnrichedRepoData if needed elsewhere
    dataclass,
    field,
)
from typing import Any, Dict, List, Optional

import litellm
from playwright.sync_api import Page, Playwright, sync_playwright
from pydantic import ValidationError
from rich import print

# Import the Pydantic models from models.py
from .models import IssueAnalysis, LLMIssueAnalysisSchema

# --- Configuration ---
# Ensure OPENROUTER_API_KEY is set in your environment for LiteLLM
# LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/google/gemini-2.5-pro-preview-03-25")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-pro-preview-03-25")
MAX_COMMENTS_FOR_LLM = 100  # Limit number of comments sent to LLM
PLAYWRIGHT_TIMEOUT = 30000  # 30 seconds for page operations


# --- Data Structures ---
# IssueAnalysis is now imported from models.py


@dataclass
class ScrapedIssueData:
    """Raw data scraped from a GitHub issue page."""

    title: Optional[str] = None
    body: Optional[str] = None
    comments: List[str] = field(default_factory=list)
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
            print("[italic grey50]Playwright browser started.[/italic grey50]")
        except Exception as e:
            print(f"[red]Error starting Playwright: {e}")
            self._cleanup()  # Ensure cleanup if launch fails
            raise  # Re-raise the exception
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        print("[italic grey50]Playwright browser stopped.[/italic grey50]")

    def _cleanup(self):
        """Safely closes browser and stops Playwright."""
        if self.browser:
            try:
                self.browser.close()
                self.browser = None
            except Exception as e:
                print(f"[yellow]Warning: Error closing Playwright browser: {e}")
        if self.playwright:
            try:
                self.playwright.stop()
                self.playwright = None
            except Exception as e:
                print(f"[yellow]Warning: Error stopping Playwright: {e}")

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
            print(f"  Scraping: {issue_url} ...")
            page.goto(issue_url, wait_until="domcontentloaded", timeout=self.timeout)
            print("    Page loaded. Waiting for container elements...")

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
            print(f"    Waiting for title container: '{title_container_selector}'")
            page.wait_for_selector(
                title_container_selector, state="attached", timeout=self.timeout
            )
            print(f"    Waiting for body container: '{body_container_selector}'")
            # Wait for the *first* comment container, which usually holds the body
            page.locator(body_container_selector).first.wait_for(
                state="attached", timeout=self.timeout
            )
            print("    Container elements attached.")

            # 2. Wait for specific title and body elements to be visible
            print(f"    Waiting for title element to be visible: '{title_selector}'")
            page.wait_for_selector(
                title_selector, state="visible", timeout=self.timeout
            )
            title_elem = page.locator(title_selector).first
            title = title_elem.text_content(timeout=self.timeout / 2) or ""
            title = title.strip()
            print(f"    Title found: '{title[:50]}...'")

            print(f"    Waiting for body element to be visible: '{body_selector}'")
            # Target the body within the *first* comment container
            body_elem = (
                page.locator(body_container_selector).first.locator(body_selector).first
            )
            body_elem.wait_for(state="visible", timeout=self.timeout)
            body = body_elem.text_content(timeout=self.timeout / 2) or ""
            body = body.strip()
            print(f"    Body found: '{body[:50]}...'")

            # 3. Get comments (using updated selectors and logic)
            print(f"    Locating comment containers: '{comment_container_selector}'")
            comments = []
            # Locate all timeline elements
            timeline_elements = page.locator(comment_container_selector)
            timeline_count = timeline_elements.count()
            print(f"    Found {timeline_count} potential timeline elements.")

            # Identify the main issue body element to skip it during comment iteration
            main_body_element = page.locator(body_container_selector).first

            for i in range(timeline_count):
                timeline_element = timeline_elements.nth(i)

                # Check if this timeline element contains the main issue body
                # We can do this by checking if the main body element is an ancestor
                # or if the timeline element itself contains the specific body selector
                # A simpler check might be to see if it contains our specific body_selector
                # and if that element matches the one we already found.
                potential_body_in_timeline = timeline_element.locator(
                    body_selector
                ).first
                is_main_body = False
                if potential_body_in_timeline.count() > 0:
                    # Check if this potential body element is the same as the main one we found earlier
                    # This requires comparing the elements directly, which can be tricky.
                    # A simpler heuristic: assume the *first* timeline element containing
                    # the body_selector is the main issue body.
                    # Let's refine: Check if the timeline element *contains* the main_body_element we located.
                    # Playwright doesn't have a direct 'contains' for elements.
                    # Alternative: Check if the timeline element's outer HTML contains the main body's outer HTML (less reliable).
                    # Let's stick to the assumption: the first timeline element containing a 'markdown-body' is the main issue.
                    # We already have the main body text. Let's find the body within this timeline element
                    # and compare its text content.

                    current_body_elem = timeline_element.locator(
                        comment_body_selector
                    ).first
                    if current_body_elem.count() > 0:
                        current_body_text = (
                            current_body_elem.text_content(timeout=self.timeout / 4)
                            or ""
                        )
                        # Compare stripped text content - might be fragile if body is empty/identical to a comment
                        if (
                            current_body_text.strip() == body
                        ):  # Compare with the main body text we scraped
                            # More robust check might be needed if bodies can be identical
                            # For now, assume the first match is the main body
                            if not comments:  # If we haven't added any comments yet, this is likely the main body
                                print(
                                    f"    Skipping timeline element {i} (identified as main issue body)."
                                )
                                is_main_body = True

                if not is_main_body:
                    # This is likely a comment, try to extract its body
                    comment_body_elem = timeline_element.locator(
                        comment_body_selector
                    ).first
                    if comment_body_elem.count() > 0:
                        text = (
                            comment_body_elem.text_content(timeout=self.timeout / 2)
                            or ""
                        )
                        text = text.strip()
                        if text:
                            print(
                                f"    Found comment in timeline element {i}: '{text[:50]}...'"
                            )
                            comments.append(text)
                        # else: # Optional: log if a comment body was found but empty
                        #    print(f"    Timeline element {i} has an empty comment body.")
                    # else: # Optional: log if a timeline element didn't contain a comment body
                    #    print(f"    Timeline element {i} did not contain a comment body selector '{comment_body_selector}'.")

            print(
                f"    Scraped: Title='{title[:50]}...', Body='{body[:50]}...', Comments={len(comments)}"
            )
            return ScrapedIssueData(title=title, body=body, comments=comments)

        except Exception as e:
            error_msg = f"Error scraping {issue_url}: {e}\n{traceback.format_exc()}"
            print(f"    [yellow]Warning: {error_msg}[/yellow]")
            return ScrapedIssueData(error=error_msg)
        finally:
            if page:
                try:
                    page.close()
                except Exception as e:
                    print(f"[yellow]Warning: Error closing page for {issue_url}: {e}")

    def _build_prompt(self, issue_data: ScrapedIssueData) -> str:
        """Builds the prompt for the LLM analysis."""
        if not issue_data.title:
            return ""  # Cannot analyze without a title

        # Limit comments sent to LLM
        comments_str = " || ".join(issue_data.comments[:MAX_COMMENTS_FOR_LLM])
        if len(issue_data.comments) > MAX_COMMENTS_FOR_LLM:
            comments_str += f" || ... (truncated, {len(issue_data.comments) - MAX_COMMENTS_FOR_LLM} more comments)"

        # Generate the schema with updated descriptions
        schema_json = json.dumps(LLMIssueAnalysisSchema.model_json_schema(), indent=2)

        prompt = f"""
Analyze the following GitHub issue content (title, body, comments) to determine its suitability as a task potentially solvable by an autonomous AI software agent. Focus on understanding the core problem, its complexity, and requirements.

**Input Issue Content:**

*   **Title:** {issue_data.title}
*   **Body:**
    ```
    {issue_data.body}
    ```
*   **Comments (selected):**
    ```
    {comments_str}
    ```

**Instructions:**

Your task is to analyze the provided GitHub issue content (title, body, comments).
Based on your analysis, generate **ONLY** a valid JSON object that strictly conforms to the following JSON schema.
**DO NOT** include any text, explanations, apologies, summaries, or markdown formatting before or after the JSON object. Your entire response must be the JSON object itself.

**Required JSON Output Schema:**

```json
{schema_json}
```

**Generate the JSON output now:**
"""
        return prompt

    def _analyze_issue_with_llm(self, issue_data: ScrapedIssueData) -> Dict[str, Any]:
        """
        Analyzes the scraped issue data using LiteLLM, expecting a JSON object
        that conforms to the LLMIssueAnalysisSchema Pydantic model.

        Returns:
            A dictionary containing the validated analysis data or an 'error' key.
        """
        prompt = self._build_prompt(issue_data)
        if not prompt:
            return {"error": "Cannot analyze issue without a title."}

        print(f"    Analyzing with LLM ({LLM_MODEL})...")
        analysis_result: Optional[LLMIssueAnalysisSchema] = None
        error_message: Optional[str] = None
        raw_llm_output: Optional[str] = None

        try:
            response = litellm.completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                # max_tokens=512,  # Adjust if needed based on schema complexity
                # temperature=0.1,  # Lower temperature for more deterministic JSON
                # Use the Pydantic model schema for structured output
                response_format=LLMIssueAnalysisSchema,
                # response_format={
                #     "type": "json_schema",
                #     "json_schema": LLMIssueAnalysisSchema.model_json_schema(),
                # },
                # Consider adding 'strict=True' if the model supports it and you want stricter validation
                # strict=True
            )
            print(f"response={response}")
            # Extract the model's JSON output string
            raw_llm_output = response.choices[0].message.content
            print(f"raw_llm_output={raw_llm_output}")

            # Parse the raw JSON string and validate against the Pydantic model
            # Assuming raw_llm_output is now clean JSON
            json_string = raw_llm_output.strip()  # Still good to strip whitespace
            llm_data = json.loads(json_string)
            analysis_result = LLMIssueAnalysisSchema.model_validate(llm_data)
            print("    LLM Analysis successful and validated.")
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
            print(f"    [red]Error: {error_message}[/red]")
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
            print(
                f"[yellow]Skipping enrichment for {repo_name}: Missing 'html_url' or valid 'found_issues' list.[/yellow]"
            )
            return analysis_results  # Return empty list

        print(f"Processing issues for repo: {repo_name}")
        for issue_number in found_issues:
            issue_url = f"{repo_html_url}/issues/{issue_number}"
            analysis: Optional[IssueAnalysis] = None  # Use the Pydantic model type
            error_str: Optional[str] = None

            if not isinstance(issue_number, int):
                error_str = f"Invalid issue number format: {issue_number}"
                print(f"  [yellow]Warning: {error_str} in {repo_name}.[/yellow]")
                # Create a minimal IssueAnalysis object with the error
                # Need dummy values for required fields if validation fails early
                # Create a minimal IssueAnalysis object with the error
                # Need dummy values for ALL required fields if validation fails early
                analysis = IssueAnalysis(
                    issue_number=issue_number,
                    issue_url=issue_url,
                    error=error_str,
                    # Provide dummy values for all required fields from LLMIssueAnalysisSchema
                    full_problem_statement="Error: Invalid issue number format.",
                    complexity_score=-1.0,
                    is_good_first_issue_for_agent=False,
                    reasoning="N/A - Invalid issue number",
                    required_domain_knowledge=-1.0,
                    codebase_familiarity_needed=-1.0,
                    scope_clarity=-1.0,
                    meta_notes="Invalid issue number provided in cache.",
                )
                analysis_results.append(analysis)
                continue

            # 1. Scrape
            scraped_data = self.scrape_github_issue(issue_url)
            if scraped_data.error:
                error_str = f"Scraping failed: {scraped_data.error}"
                # Create analysis object with error, providing dummy required fields
                # Create analysis object with error, providing dummy required fields
                analysis = IssueAnalysis(
                    issue_number=issue_number,
                    issue_url=issue_url,
                    error=error_str,
                    # Provide dummy values for all required fields from LLMIssueAnalysisSchema
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
                continue  # Move to next issue

            # 2. Analyze
            llm_result_dict = self._analyze_issue_with_llm(scraped_data)

            # 3. Validate and Create Pydantic Object
            if llm_result_dict.get("error"):
                error_str = f"LLM analysis failed: {llm_result_dict['error']}"
                # Create analysis object with error, providing dummy required fields
                # Create analysis object with error, providing dummy required fields
                analysis = IssueAnalysis(
                    issue_number=issue_number,
                    issue_url=issue_url,
                    error=error_str,
                    # Provide dummy values for all required fields from LLMIssueAnalysisSchema
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
                    print(f"    [red]Error: {error_str}[/red]")
                    # Create analysis object with error, providing dummy required fields
                    # Create analysis object with error, providing dummy required fields
                    analysis = IssueAnalysis(
                        issue_number=issue_number,
                        issue_url=issue_url,
                        error=error_str,
                        # Provide dummy values for all required fields from LLMIssueAnalysisSchema
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

        print(f"Finished processing {len(found_issues)} issue(s) for {repo_name}.")
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
        print(f"[red]Error processing repo {repo_name} in parallel worker: {e}")
        # Add an error marker to the output data for this repo
        enriched_data["enrichment_error"] = str(e)

    enriched_data["issue_analysis"] = analysis_list
    return enriched_data
