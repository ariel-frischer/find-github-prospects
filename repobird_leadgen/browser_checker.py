from __future__ import annotations

# from rich.console import Console # Remove Console import
import logging  # Import logging
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple  # Added Dict, Any

from playwright.sync_api import (
    Error as PlaywrightError,
)
from playwright.sync_api import (
    Page,
    Playwright,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
)

logger = logging.getLogger(__name__)  # Get logger

# --- Constants moved from filter_issues_by_prs.py ---
# Selector for the sidebar section containing "Development"
DEV_SECTION_SELECTOR = (
    'div[data-testid="sidebar-section"]:has(h3:has-text("Development"))'
)
# Selector for the UL containing linked PRs within the Development section
LINKED_PR_LIST_SELECTOR = (
    f"{DEV_SECTION_SELECTOR} ul[data-testid='issue-viewer-linked-pr-container']"
)
# Selector for list items (actual PR links) within the container
LINKED_PR_ITEM_SELECTOR = f"{LINKED_PR_LIST_SELECTOR} li"  # Keep this selector definition for context, though the function uses a more specific one


# --- Function moved from filter_issues_by_prs.py ---
def check_dev_section_for_prs(page: Page, issue_url: str) -> bool:
    """
    Uses Playwright to check the GitHub issue page for linked PRs in the
    'Development' sidebar section. Assumes page is already navigated to issue_url.

    Args:
        page: The Playwright Page object to use.
        issue_url: The URL of the GitHub issue (used for logging).

    Returns:
        True if linked PRs are found in the Development section, False otherwise.
        Returns False on Playwright errors (e.g., timeout).
    """
    logger.info(f"Checking Development section for linked PRs on {issue_url}...")
    try:
        # Wait for the Development section header itself to be visible first
        page.locator(f'{DEV_SECTION_SELECTOR} h3:has-text("Development")').wait_for(
            state="visible",
            timeout=15000,  # Shorter timeout for this check
        )
        # console.print("        - Development section header found.") # Debug

        # Check specifically for anchor tags with '/pull/' in href within the list
        # This covers open, closed, and merged PRs listed in the Development section
        linked_pr_links = page.locator(f"{LINKED_PR_LIST_SELECTOR} a[href*='/pull/']")
        pr_count = linked_pr_links.count()

        if pr_count > 0:
            logger.info(
                f"Found {pr_count} linked PR(s) in Development section for {issue_url}."
            )
            return True  # Found one or more linked PRs
        else:
            # Section header found, but no PR links within the specific list structure
            logger.info(
                f"Development section found, but no linked PRs detected within it for {issue_url}."
            )
            return False

    except PlaywrightTimeoutError:
        # If the Development section header doesn't appear, assume no linked PRs shown
        logger.warning(
            f"Timeout waiting for Development section on {issue_url}. Assuming no linked PRs."
        )
        return False
    except PlaywrightError as e:
        logger.error(
            f"Playwright error checking Dev section on {issue_url}: {e}. Assuming no linked PRs."
        )
        return False
    except Exception as e:
        logger.exception(f"Unexpected error checking Dev section on {issue_url}: {e}")
        # traceback.print_exc() # logger.exception includes traceback
        return False


class BrowserIssueChecker:
    """Uses Playwright to check GitHub frontend for open issues with a specific label."""

    # More specific selectors might be needed if GitHub UI changes
    _ISSUE_FILTER_INPUT_SELECTOR = (
        "input#repository-input"  # Updated selector based on inspection
    )
    # Use data-testid for issue rows
    _ISSUE_ROW_SELECTOR = "div[data-testid='issue-row']"
    # Use data-testid for the issue link/title within a row
    _ISSUE_LINK_SELECTOR = "a[data-testid='issue-title-link']"
    # Selector for the 'no results' message container (seems stable enough)
    _NO_RESULTS_SELECTOR = "div.blankslate"
    # Regex to extract issue number from the href of the issue link
    # Regex to extract issue number from the href of the issue link
    # Regex to extract issue number from the href of the issue link
    _ISSUE_ID_RE = re.compile(r"/issues/(\d+)$")
    # Selectors for individual issue page details
    # Target the BDI element containing the title text using its data-testid
    _ISSUE_TITLE_SELECTOR = 'bdi[data-testid="issue-title"]'
    _ISSUE_BODY_SELECTOR = 'div.markdown-body[data-testid="markdown-body"]'
    # Selectors for Development section PR check
    _DEV_SECTION_SELECTOR = (
        'div[data-testid="sidebar-section"]:has(h3:has-text("Development"))'
    )
    _LINKED_PR_LIST_SELECTOR = (
        f"{_DEV_SECTION_SELECTOR} ul[data-testid='issue-viewer-linked-pr-container']"
    )
    _LINKED_PR_ITEM_SELECTOR = f"{_LINKED_PR_LIST_SELECTOR} li"

    # Increase default timeout significantly for debugging
    def __init__(
        self, headless: bool = True, check_delay: float = 2.0, timeout: int = 45000
    ):
        """
        Initializes the checker.

        Args:
            headless: Run the browser in headless mode.
            check_delay: Seconds to wait between checks to avoid detection.
            timeout: Default timeout in milliseconds for Playwright actions.
        """
        self.headless = headless
        self.check_delay = check_delay
        self.timeout = timeout  # milliseconds
        self._playwright: Optional[Playwright] = None
        self._browser = None
        logger.info(
            f"BrowserChecker Initialized (Headless: {headless}, Delay: {check_delay}s, Timeout: {timeout}ms)"
        )

    def __enter__(self) -> BrowserIssueChecker:
        """Starts Playwright and launches the browser."""
        try:
            self._playwright = sync_playwright().start()
            # Launch browser - consider adding slow_mo for debugging if needed
            # self._browser = self._playwright.chromium.launch(headless=self.headless, slow_mo=50)
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            logger.info("BrowserChecker: Playwright started and browser launched.")
        except Exception as e:
            logger.exception(f"BrowserChecker: Error starting Playwright/Browser: {e}")
            logger.error(
                "Ensure browser binaries are installed ('playwright install')."
            )
            self._cleanup()  # Ensure cleanup if launch fails
            raise  # Re-raise the exception
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the browser and stops Playwright."""
        self._cleanup()
        logger.info("BrowserChecker: Browser closed and Playwright stopped.")

    def _cleanup(self):
        """Safely closes browser and stops Playwright."""
        if self._browser:
            try:
                self._browser.close()
            except Exception as e:
                logger.warning(f"BrowserChecker: Error closing browser: {e}")
            self._browser = None
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception as e:
                logger.warning(f"BrowserChecker: Error stopping Playwright: {e}")
            self._playwright = None

    def _fetch_issue_details(
        self, page: Page, issue_url: str
    ) -> Optional[Dict[str, str]]:
        """Navigates to an issue URL and extracts title and description."""
        logger.info(f"Fetching details for: {issue_url}")
        try:
            page.goto(issue_url, wait_until="domcontentloaded", timeout=self.timeout)
            # Wait for container elements to be attached, give full timeout
            # Use simpler container selectors for waiting
            title_container_selector = 'div[data-component="TitleArea"]'
            # Use data-testid for the body container
            body_container_selector = 'div[data-testid="issue-body"]'

            # console.print(f"      Waiting for title container: '{title_container_selector}'...")
            page.wait_for_selector(
                title_container_selector, state="attached", timeout=self.timeout
            )
            # console.print("      Title container found.") # Commented out

            # console.print(f"      Waiting for body container: '{body_container_selector}'...") # Commented out
            page.wait_for_selector(
                body_container_selector, state="attached", timeout=self.timeout
            )
            # console.print("      Body container found.") # Commented out

            # Explicitly wait for the title element (BDI) to be VISIBLE
            # console.print(f"      Waiting for title element BDI to be visible: '{self._ISSUE_TITLE_SELECTOR}'...")
            page.wait_for_selector(
                self._ISSUE_TITLE_SELECTOR, state="visible", timeout=self.timeout
            )
            # console.print("      Title element BDI is visible.")
            title_element = page.locator(self._ISSUE_TITLE_SELECTOR).first

            # Explicitly wait for the body element to be VISIBLE
            # console.print(f"      Waiting for body element to be visible: '{self._ISSUE_BODY_SELECTOR}'...") # Commented out
            page.wait_for_selector(
                self._ISSUE_BODY_SELECTOR, state="visible", timeout=self.timeout
            )
            # console.print("      Body element is visible.") # Commented out
            body_element = page.locator(self._ISSUE_BODY_SELECTOR).first
            # console.print("      Body element located.") # Commented out

            # Revert to text_content() with a shorter timeout for the extraction itself
            # console.print("      Extracting title text using text_content()...") # Commented out
            title = title_element.text_content(timeout=self.timeout / 2) or ""
            # console.print("      Title text extracted.") # Commented out

            # console.print("      Extracting body text using text_content()...") # Commented out
            description = body_element.text_content(timeout=self.timeout / 2) or ""
            # console.print("      Body text extracted.") # Commented out

            # Basic cleanup
            title = title.strip()
            description = description.strip()

            title = title.strip()
            description = description.strip()

            logger.info(f"Fetched Title: '{title[:50]}...' for {issue_url}")
            return {"title": title, "description": description}

        except PlaywrightTimeoutError:
            logger.warning(f"Timeout fetching details for {issue_url}")
            return None
        except PlaywrightError as pe:
            logger.warning(f"Playwright error fetching details for {issue_url}: {pe}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error fetching details for {issue_url}: {e}")
            return None

    def _check_issue_for_linked_prs(self, page: Page, issue_url: str) -> bool:
        """
        Checks the specific issue page's 'Development' section for linked PRs.
        Assumes the page is already navigated to the issue_url.

        Args:
            page: The Playwright Page object (already navigated).
            issue_url: The URL of the issue (used for logging).

        Returns:
            True if linked PRs are found in the Development section, False otherwise.
        """
        # Call the standalone function, passing the page and URL
        # The standalone function handles its own logging and exceptions.
        return check_dev_section_for_prs(page, issue_url)

    def _get_new_page(self) -> Page:
        """Creates a new browser page."""
        if not self._browser:
            raise RuntimeError("Browser is not launched. Use as a context manager.")
        try:
            # Use a new context for better isolation if needed, but page is simpler for now
            page = self._browser.new_page()
            page.set_default_timeout(self.timeout)
            return page
        except Exception as e:
            logger.exception(f"BrowserChecker: Error creating new browser page: {e}")
            raise

    def check_repo_for_issue_label(
        self, repo_full_name: str, label: str, max_linked_prs: Optional[int] = None
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Checks a repository's issues page via Playwright for open issues matching the query.
        If issues are found, fetches their number, title, and description.

        Args:
            repo_full_name: The 'owner/repo' name string.
            label: The issue label to include in the search query.
            max_linked_prs: If 0, issues with linked PRs in the Development section
                            will be excluded. If None or other int, this check is skipped.

        Returns:
            A tuple: (bool indicating if issues were found, list of issue detail dicts).
            Each dict contains: {'number': int, 'title': str, 'description': str}
        """
        if not self._browser or not self._playwright:
            logger.error(
                "BrowserChecker: Playwright/Browser not initialized. Cannot perform check."
            )
            return False, []  # Return tuple (bool, list)

        page = None
        # Construct the raw query string, including no:assignee and no:parent-issue
        raw_query = f'is:issue state:open label:"{label}" no:assignee no:parent-issue'
        # URL encode the *entire* query string once using quote_plus for query parameters
        encoded_query = urllib.parse.quote_plus(raw_query)
        # Construct the final URL
        filtered_issues_url = (
            f"https://github.com/{repo_full_name}/issues?q={encoded_query}"
        )

        # found_issue_flag = False # This was the unused variable, removed.
        issue_numbers: List[int] = []
        found_issues_details: List[Dict[str, Any]] = []  # Store full details

        try:
            logger.info(f"BrowserChecker: Getting new page for {repo_full_name}...")
            page = self._get_new_page()
            logger.info(
                f"BrowserChecker: Navigating directly to filtered URL: {filtered_issues_url}..."
            )
            # Navigate directly to the pre-filtered URL
            page.goto(
                filtered_issues_url, wait_until="domcontentloaded", timeout=self.timeout
            )
            logger.info(
                f"BrowserChecker: Navigation to {filtered_issues_url} successful."
            )

            # No need to interact with the filter input anymore

            # Wait for the results to potentially update. We need to wait for either:
            # 1. Issue rows to appear (meaning matches found)
            # 2. The "No results" message to appear
            # 3. A timeout (assume no results or error)

            # Wait for either the first issue row OR the no results message to be attached
            results_selector = (
                f"{self._ISSUE_ROW_SELECTOR}, {self._NO_RESULTS_SELECTOR}"
            )
            logger.info(
                f"BrowserChecker: Waiting for results element '{results_selector}' to be attached..."
            )
            try:
                # Wait for either the first issue row OR the no results blankslate to be attached
                page.wait_for_selector(
                    results_selector,
                    state="attached",  # Wait for element to be in DOM, not necessarily visible
                    timeout=self.timeout,
                )
                logger.info(
                    "BrowserChecker: Results element (issue row or blankslate) attached."
                )

                # Now check specifically which element was found and if issue rows exist
                # Use the updated row selector directly
                issue_rows = page.locator(self._ISSUE_ROW_SELECTOR)
                count = issue_rows.count()
                logger.info(
                    f"BrowserChecker: Found {count} potential issue row(s) matching selector '{self._ISSUE_ROW_SELECTOR}'."
                )

                if count > 0:
                    # Extract issue numbers from links within rows
                    for i in range(count):
                        row = issue_rows.nth(i)
                        # Find the specific link within the row
                        link_element = row.locator(self._ISSUE_LINK_SELECTOR).first
                        href = link_element.get_attribute("href")
                        if href:
                            match = self._ISSUE_ID_RE.search(href)
                            if match:
                                try:
                                    issue_num = int(match.group(1))
                                    issue_numbers.append(issue_num)
                                except (ValueError, IndexError):
                                    logger.warning(
                                        f"Could not parse issue number from href: {href}"
                                    )
                            else:
                                logger.warning(
                                    f"Could not find issue number pattern in href: {href}"
                                )
                        else:
                            logger.warning(
                                f"Issue link element found in row {i} but has no href attribute."
                            )
                    logger.info(
                        f"BrowserChecker: Extracted issue numbers: {issue_numbers}"
                    )

                    # --- Fetch details for each found issue ---
                    if issue_numbers:
                        logger.info(
                            f"Fetching details for {len(issue_numbers)} issues..."
                        )
                        # Reuse the same page for fetching details
                        for issue_num in issue_numbers:
                            issue_url = f"https://github.com/{repo_full_name}/issues/{issue_num}"
                            details = self._fetch_issue_details(page, issue_url)
                            if details:
                                found_issues_details.append(
                                    {
                                        "number": issue_num,
                                        "title": details["title"],
                                        "description": details["description"],
                                    }
                                )

                                # --- Check for linked PRs if max_linked_prs is 0 ---
                                if max_linked_prs == 0:
                                    # Reuse the page, which is already at the issue URL
                                    has_prs = self._check_issue_for_linked_prs(
                                        page, issue_url
                                    )
                                    if has_prs:
                                        # Add explicit message about exclusion
                                        logger.warning(
                                            f"Issue #{issue_num} has linked PRs. Excluding from results (max_linked_prs=0)."
                                        )
                                        # Remove the details we just added
                                        found_issues_details.pop()
                                    else:
                                        logger.info(
                                            f"Issue #{issue_num} has no linked PRs. Keeping."
                                        )
                                # --- End PR check ---

                            else:  # Details fetch failed
                                logger.warning(
                                    f"Skipping details and PR check for issue {issue_num} due to fetch error."
                                )
                            time.sleep(0.5)  # Small delay between issue detail fetches
                        logger.info(
                            "Finished fetching details and checking PRs (if applicable)."
                        )

                else:  # No issue rows found initially
                    # No issue rows found, check if the "no results" message is visible
                    no_results_element = page.locator(self._NO_RESULTS_SELECTOR)
                    if no_results_element.is_visible():
                        logger.info("BrowserChecker: 'No results' message is visible.")
                    else:
                        logger.warning(
                            "BrowserChecker: Neither issue rows nor 'no results' message were visible after waiting."
                        )

            except PlaywrightTimeoutError:
                logger.warning(
                    f"BrowserChecker: Timeout waiting for issue results container for {repo_full_name}. Assuming no matching issues."
                )
                # No action needed on flags, handled by final check
            except PlaywrightError as pe:
                logger.warning(
                    f"BrowserChecker: Playwright error checking results for {repo_full_name}: {pe}. Assuming no matching issues."
                )
                # No action needed on flags, handled by final check

        except PlaywrightTimeoutError:
            logger.error(
                f"BrowserChecker: Timeout error during navigation or interaction for {repo_full_name}."
            )
            # No action needed on flags, handled by final check
        except PlaywrightError as pe:
            logger.error(
                f"BrowserChecker: Playwright error during check for {repo_full_name}: {pe}"
            )
            # No action needed on flags, handled by final check
        except Exception as e:
            logger.exception(
                f"BrowserChecker: Unexpected error checking {repo_full_name}: {e}"
            )
            # No action needed on flags, handled by final check
        finally:
            if page:
                try:
                    page.close()
                except Exception as e:
                    logger.warning(f"BrowserChecker: Error closing page: {e}")
            # Add delay regardless of success/failure to slow down requests
            # logger.debug(f"BrowserChecker: Waiting {self.check_delay:.1f}s before next check...")
            time.sleep(self.check_delay)

        # Return True only if the *final* list of details is non-empty
        final_found_flag = bool(found_issues_details)
        return final_found_flag, found_issues_details

    def scrape_readme_text(self, repo_url: str) -> Optional[str]:
        """
        Navigates to the repository URL and scrapes the text content of the README.

        Args:
            repo_url: The full URL to the repository's main page.

        Returns:
            The text content of the README, or None if not found or an error occurs.
        """
        if not self._browser or not self._playwright:
            logger.error(
                "BrowserChecker: Playwright/Browser not initialized. Cannot scrape README."
            )
            return None

        page = None
        readme_selector = "article.markdown-body"  # Selector for the README content

        try:
            logger.info(f"BrowserChecker: Getting new page for README: {repo_url}...")
            page = self._get_new_page()
            logger.info(f"BrowserChecker: Navigating to {repo_url}...")
            page.goto(repo_url, wait_until="domcontentloaded", timeout=self.timeout)
            logger.info("BrowserChecker: Navigation successful.")

            logger.info(
                f"BrowserChecker: Waiting for README selector: '{readme_selector}'..."
            )
            # Wait for the README container to be visible
            readme_element = page.locator(readme_selector).first
            readme_element.wait_for(state="visible", timeout=self.timeout)
            logger.info("BrowserChecker: README element located.")

            # Extract text content
            readme_text = readme_element.text_content(timeout=self.timeout / 2)
            logger.info(
                f"BrowserChecker: README text extracted (length: {len(readme_text or '')})."
            )
            return readme_text.strip() if readme_text else None

        except PlaywrightTimeoutError:
            logger.warning(f"BrowserChecker: Timeout waiting for README on {repo_url}")
            return None
        except PlaywrightError as pe:
            logger.error(
                f"BrowserChecker: Playwright error scraping README from {repo_url}: {pe}"
            )
            return None
        except Exception as e:
            logger.exception(
                f"BrowserChecker: Unexpected error scraping README from {repo_url}: {e}"
            )
            # traceback.print_exc() # logger.exception includes traceback
            return None
        finally:
            if page:
                try:
                    page.close()
                except Exception as e:
                    logger.warning(
                        f"BrowserChecker: Error closing page after README scrape: {e}"
                    )
            # Add delay? Maybe not needed for a single scrape like this unless called rapidly.
            # time.sleep(self.check_delay)


# Example Usage (for testing)
if __name__ == "__main__":
    # repo_to_check = "octocat/Spoon-Knife" # A repo likely without 'good first issue'
    # label_to_check = "good first issue"

    repo_to_check = "microsoft/vscode"  # A repo likely with 'good first issue'
    label_to_check = "good first issue"

    # repo_to_check = "facebook/react"
    # label_to_check = "good first issue" # Check a specific label

    checker = BrowserIssueChecker(
        headless=True,
        check_delay=1,
        timeout=30000,  # Increased timeout slightly
    )
    with checker:  # Use context manager
        has_issues, issue_details = (
            checker.check_repo_for_issue_label(  # Renamed variable here
                repo_to_check,
                label_to_check,
                max_linked_prs=0,  # Test the PR exclusion
            )
        )
        # Use standard print for example output, or configure logging for example run
        print(
            f"\nCheck result for {repo_to_check} with label '{label_to_check}' (excluding linked PRs): Found={has_issues}"
        )
        if issue_details:
            print("  Issue Details:")
            for issue in issue_details[:3]:  # Print details for first few
                print(f"    - Number: {issue['number']}")
                print(f"      Title: {issue['title'][:80]}...")
                # print(f"      Desc: {issue['description'][:100]}...") # Can be long
        elif has_issues:
            print("  (Issues found, but details could not be fetched or all had PRs)")

        # Test another one (e.g., a repo likely without the label)
        repo_to_check_neg = "octocat/Spoon-Knife"
        has_issues_neg, issue_details_neg = checker.check_repo_for_issue_label(
            repo_to_check_neg, label_to_check
        )
        print(
            f"\nCheck result for {repo_to_check_neg} with label '{label_to_check}': Found={has_issues_neg}"
        )
        if issue_details_neg:
            print("  Issue Details:", issue_details_neg)

        # Test another label on the first repo
        # result = checker.check_repo_for_issue_label("microsoft/vscode", "help wanted")
        # print(f"\nCheck result for microsoft/vscode with label 'help wanted': {result}")

        # --- Test README Scraping ---
        print("\n" + "=" * 20 + " Testing README Scraping " + "=" * 20)
        # repo_to_scrape = "https://github.com/aio-libs/aiohttp"
        repo_to_scrape = "https://github.com/pytorch/executorch"  # Use the example repo
        readme_content = checker.scrape_readme_text(repo_to_scrape)
        if readme_content:
            print(f"\nSuccessfully scraped README from {repo_to_scrape}:")
            print(f"Length: {len(readme_content)}")
            print("-" * 10 + " Start of README " + "-" * 10)
            print(readme_content[:1000] + "...")  # Print first 1000 chars
            print("-" * 10 + " End of README " + "-" * 10)
        else:
            print(f"\nFailed to scrape README from {repo_to_scrape}")
