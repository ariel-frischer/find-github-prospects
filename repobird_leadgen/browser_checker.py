from __future__ import annotations

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
from rich import print


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
        print(
            f"[BrowserChecker] Initialized (Headless: {headless}, Delay: {check_delay}s, Timeout: {timeout}ms)"
        )

    def __enter__(self) -> BrowserIssueChecker:
        """Starts Playwright and launches the browser."""
        try:
            self._playwright = sync_playwright().start()
            # Launch browser - consider adding slow_mo for debugging if needed
            # self._browser = self._playwright.chromium.launch(headless=self.headless, slow_mo=50)
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            print("[BrowserChecker] Playwright started and browser launched.")
        except Exception as e:
            print(f"[red][BrowserChecker] Error starting Playwright/Browser: {e}")
            print("[red]Ensure browser binaries are installed ('playwright install').")
            self._cleanup()  # Ensure cleanup if launch fails
            raise  # Re-raise the exception
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the browser and stops Playwright."""
        self._cleanup()
        print("[BrowserChecker] Browser closed and Playwright stopped.")

    def _cleanup(self):
        """Safely closes browser and stops Playwright."""
        if self._browser:
            try:
                self._browser.close()
            except Exception as e:
                print(f"[yellow][BrowserChecker] Error closing browser: {e}")
            self._browser = None
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception as e:
                print(f"[yellow][BrowserChecker] Error stopping Playwright: {e}")
            self._playwright = None

    def _fetch_issue_details(
        self, page: Page, issue_url: str
    ) -> Optional[Dict[str, str]]:
        """Navigates to an issue URL and extracts title and description."""
        print(f"    Fetching details for: {issue_url}")
        try:
            page.goto(issue_url, wait_until="domcontentloaded", timeout=self.timeout)
            # Wait for container elements to be attached, give full timeout
            # Use simpler container selectors for waiting
            title_container_selector = 'div[data-component="TitleArea"]'
            # Use data-testid for the body container
            body_container_selector = 'div[data-testid="issue-body"]'

            # print(f"      Waiting for title container: '{title_container_selector}'...")
            page.wait_for_selector(
                title_container_selector, state="attached", timeout=self.timeout
            )
            # print("      Title container found.") # Commented out

            # print(f"      Waiting for body container: '{body_container_selector}'...") # Commented out
            page.wait_for_selector(
                body_container_selector, state="attached", timeout=self.timeout
            )
            # print("      Body container found.") # Commented out

            # Explicitly wait for the title element (BDI) to be VISIBLE
            # print(f"      Waiting for title element BDI to be visible: '{self._ISSUE_TITLE_SELECTOR}'...")
            page.wait_for_selector(
                self._ISSUE_TITLE_SELECTOR, state="visible", timeout=self.timeout
            )
            # print("      Title element BDI is visible.")
            title_element = page.locator(self._ISSUE_TITLE_SELECTOR).first

            # Explicitly wait for the body element to be VISIBLE
            # print(f"      Waiting for body element to be visible: '{self._ISSUE_BODY_SELECTOR}'...") # Commented out
            page.wait_for_selector(
                self._ISSUE_BODY_SELECTOR, state="visible", timeout=self.timeout
            )
            # print("      Body element is visible.") # Commented out
            body_element = page.locator(self._ISSUE_BODY_SELECTOR).first
            # print("      Body element located.") # Commented out

            # Revert to text_content() with a shorter timeout for the extraction itself
            # print("      Extracting title text using text_content()...") # Commented out
            title = title_element.text_content(timeout=self.timeout / 2) or ""
            # print("      Title text extracted.") # Commented out

            # print("      Extracting body text using text_content()...") # Commented out
            description = body_element.text_content(timeout=self.timeout / 2) or ""
            # print("      Body text extracted.") # Commented out

            # Basic cleanup
            title = title.strip()
            description = description.strip()

            print(f"      Title: '{title[:50]}...'")
            return {"title": title, "description": description}

        except PlaywrightTimeoutError:
            print(f"      [yellow]Timeout fetching details for {issue_url}")
            return None
        except PlaywrightError as pe:
            print(
                f"      [yellow]Playwright error fetching details for {issue_url}: {pe}"
            )
            return None
        except Exception as e:
            print(f"      [red]Unexpected error fetching details for {issue_url}: {e}")
            return None

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
            print(f"[red][BrowserChecker] Error creating new browser page: {e}")
            raise

    def check_repo_for_issue_label(
        self, repo_full_name: str, label: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Checks a repository's issues page via Playwright for open issues matching the query.
        If issues are found, fetches their number, title, and description.

        Args:
            repo_full_name: The 'owner/repo' name string.
            label: The issue label to include in the search query.

        Returns:
            A tuple: (bool indicating if issues were found, list of issue detail dicts).
            Each dict contains: {'number': int, 'title': str, 'description': str}
        """
        if not self._browser or not self._playwright:
            print(
                "[red][BrowserChecker] Playwright/Browser not initialized. Cannot perform check."
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

        found_issue_flag = False
        issue_numbers: List[int] = []
        found_issues_details: List[Dict[str, Any]] = []  # Store full details

        try:
            print(f"[BrowserChecker] Getting new page for {repo_full_name}...")
            page = self._get_new_page()
            print(
                f"[BrowserChecker] Navigating directly to filtered URL: {filtered_issues_url}..."
            )
            # Navigate directly to the pre-filtered URL
            page.goto(
                filtered_issues_url, wait_until="domcontentloaded", timeout=self.timeout
            )
            print(f"[BrowserChecker] Navigation to {filtered_issues_url} successful.")

            # No need to interact with the filter input anymore

            # Wait for the results to potentially update. We need to wait for either:
            # 1. Issue rows to appear (meaning matches found)
            # 2. The "No results" message to appear
            # 3. A timeout (assume no results or error)

            # Wait for either the first issue row OR the no results message to be attached
            results_selector = (
                f"{self._ISSUE_ROW_SELECTOR}, {self._NO_RESULTS_SELECTOR}"
            )
            print(
                f"[BrowserChecker] Waiting for results element '{results_selector}' to be attached..."
            )
            try:
                # Wait for either the first issue row OR the no results blankslate to be attached
                page.wait_for_selector(
                    results_selector,
                    state="attached",  # Wait for element to be in DOM, not necessarily visible
                    timeout=self.timeout,
                )
                print(
                    "[BrowserChecker] Results element (issue row or blankslate) attached."
                )

                # Now check specifically which element was found and if issue rows exist
                # Use the updated row selector directly
                issue_rows = page.locator(self._ISSUE_ROW_SELECTOR)
                count = issue_rows.count()
                print(
                    f"[BrowserChecker] Found {count} potential issue row(s) matching selector '{self._ISSUE_ROW_SELECTOR}'."
                )

                if count > 0:
                    found_issue_flag = True
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
                                    print(
                                        f"[yellow]Could not parse issue number from href: {href}"
                                    )
                            else:
                                print(
                                    f"[yellow]Could not find issue number pattern in href: {href}"
                                )
                        else:
                            print(
                                f"[yellow]Issue link element found in row {i} but has no href attribute."
                            )
                    print(f"[BrowserChecker] Extracted issue numbers: {issue_numbers}")

                    # --- Fetch details for each found issue ---
                    if issue_numbers:
                        print(f"  Fetching details for {len(issue_numbers)} issues...")
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
                            else:
                                print(
                                    f"    [yellow]Skipping details for issue {issue_num} due to fetch error."
                                )
                            time.sleep(0.5)  # Small delay between issue detail fetches
                        print("  Finished fetching details.")

                else:
                    # No issue rows found, check if the "no results" message is visible
                    no_results_element = page.locator(self._NO_RESULTS_SELECTOR)
                    if no_results_element.is_visible():
                        print("[BrowserChecker] 'No results' message is visible.")
                    else:
                        print(
                            "[yellow][BrowserChecker] Neither issue rows nor 'no results' message were visible after waiting."
                        )
                    found_issue_flag = False  # Ensure flag is False

            except PlaywrightTimeoutError:
                print(
                    f"[yellow][BrowserChecker] Timeout waiting for issue results container for {repo_full_name}. Assuming no matching issues."
                )
                found_issue_flag = False
            except PlaywrightError as pe:
                print(
                    f"[yellow][BrowserChecker] Playwright error checking results for {repo_full_name}: {pe}. Assuming no matching issues."
                )
                found_issue_flag = False

        except PlaywrightTimeoutError:
            print(
                f"[red][BrowserChecker] Timeout error during navigation or interaction for {repo_full_name}."
            )
            found_issue_flag = False
        except PlaywrightError as pe:
            print(
                f"[red][BrowserChecker] Playwright error during check for {repo_full_name}: {pe}"
            )
            found_issue_flag = False
        except Exception as e:
            print(
                f"[red][BrowserChecker] Unexpected error checking {repo_full_name}: {e}"
            )
            found_issue_flag = False
        finally:
            if page:
                try:
                    page.close()
                except Exception as e:
                    print(f"[yellow][BrowserChecker] Error closing page: {e}")
            # Add delay regardless of success/failure to slow down requests
            # print(f"[BrowserChecker] Waiting {self.check_delay:.1f}s before next check...")
            time.sleep(self.check_delay)

        # Return the flag and the list of issue details (will be empty if flag is False)
        return found_issue_flag, found_issues_details  # Return details list


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
                repo_to_check, label_to_check
            )
        )
        print(
            f"\nCheck result for {repo_to_check} with label '{label_to_check}': Found={has_issues}"
        )
        if issue_details:
            print("  Issue Details:")
            for issue in issue_details[:3]:  # Print details for first few
                print(f"    - Number: {issue['number']}")
                print(f"      Title: {issue['title'][:80]}...")
                # print(f"      Desc: {issue['description'][:100]}...") # Can be long
        elif has_issues:
            print("  (Issues found, but details could not be fetched)")

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
