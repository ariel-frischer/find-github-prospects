from __future__ import annotations

import time
from typing import Optional

from playwright.sync_api import Page, Playwright, sync_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from rich import print


class BrowserIssueChecker:
    """Uses Playwright to check GitHub frontend for open issues with a specific label."""

    # More specific selectors might be needed if GitHub UI changes
    _ISSUE_FILTER_INPUT_SELECTOR = "input#repository-input" # Updated selector based on inspection
    _ISSUE_ROW_SELECTOR = "div.js-navigation-container div.js-issue-row" # Selector for issue list items
    _NO_RESULTS_SELECTOR = "div.blankslate" # Selector for the 'no results' message container

    # Increase default timeout significantly for debugging
    def __init__(self, headless: bool = True, check_delay: float = 2.0, timeout: int = 45000):
        """
        Initializes the checker.

        Args:
            headless: Run the browser in headless mode.
            check_delay: Seconds to wait between checks to avoid detection.
            timeout: Default timeout in milliseconds for Playwright actions.
        """
        self.headless = headless
        self.check_delay = check_delay
        self.timeout = timeout # milliseconds
        self._playwright: Optional[Playwright] = None
        self._browser = None
        print(f"[BrowserChecker] Initialized (Headless: {headless}, Delay: {check_delay}s, Timeout: {timeout}ms)")

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
            self._cleanup() # Ensure cleanup if launch fails
            raise # Re-raise the exception
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

    def check_repo_for_issue_label(self, repo_full_name: str, label: str) -> bool:
        """
        Checks a repository's issues page via Playwright for open issues with the label.

        Args:
            repo_full_name: The 'owner/repo' name string.
            label: The issue label to search for.

        Returns:
            True if at least one matching open issue is found, False otherwise.
        """
        if not self._browser or not self._playwright:
             print("[red][BrowserChecker] Playwright/Browser not initialized. Cannot perform check.")
             return False

        page = None
        issues_url = f"https://github.com/{repo_full_name}/issues"
        # Construct the filter query string GitHub uses in its UI
        filter_query = f'is:issue is:open label:"{label}"'
        found_issue = False

        try:
            print(f"[BrowserChecker] Getting new page for {repo_full_name}...")
            page = self._get_new_page()
            print(f"[BrowserChecker] Navigating to {issues_url}...")
            page.goto(issues_url, wait_until="domcontentloaded", timeout=self.timeout) # Apply timeout here too
            print(f"[BrowserChecker] Navigation to {issues_url} successful.")

            # Wait for the filter input to be visible and interactable
            print(f"[BrowserChecker] Waiting for filter input '{self._ISSUE_FILTER_INPUT_SELECTOR}'...")
            filter_input = page.locator(self._ISSUE_FILTER_INPUT_SELECTOR)
            filter_input.wait_for(state="visible", timeout=self.timeout)
            print(f"[BrowserChecker] Filter input found.")

            print(f"[BrowserChecker] Filling filter with: '{filter_query}'...")
            filter_input.fill(filter_query)
            print(f"[BrowserChecker] Filter filled.")

            print(f"[BrowserChecker] Pressing Enter on filter...")
            filter_input.press("Enter")
            print(f"[BrowserChecker] Enter pressed.")

            # Wait for the results to potentially update. We need to wait for either:
            # 1. Issue rows to appear (meaning matches found)
            # 2. The "No results" message to appear
            # 3. A timeout (assume no results or error)
            results_selector = f"{self._ISSUE_ROW_SELECTOR}, {self._NO_RESULTS_SELECTOR}"
            print(f"[BrowserChecker] Waiting for results selector '{results_selector}'...")
            try:
                # Wait for either issue rows OR the no results blankslate
                page.wait_for_selector(
                    results_selector,
                    state="attached", # Try 'attached' instead of 'visible' initially
                    timeout=self.timeout # Use the increased timeout
                )
                print(f"[BrowserChecker] Results selector found.")

                # Check if any issue rows are present *after* waiting
                print(f"[BrowserChecker] Counting issue rows ('{self._ISSUE_ROW_SELECTOR}')...")
                issue_rows = page.locator(self._ISSUE_ROW_SELECTOR)
                count = issue_rows.count() # Get count once
                if count > 0:
                    print(f"[BrowserChecker] Found {count} issue row(s) matching filter.")
                    found_issue = True
                else:
                    # Check if the "no results" message is visible
                    print(f"[BrowserChecker] No issue rows found. Checking for 'no results' message ('{self._NO_RESULTS_SELECTOR}')...")
                    no_results_element = page.locator(self._NO_RESULTS_SELECTOR)
                    if no_results_element.is_visible(): # Check visibility here
                         print("[BrowserChecker] 'No results' message is visible.")
                    else:
                         # This case is unlikely if the wait_for_selector worked, but possible
                         print("[yellow][BrowserChecker] Results selector found, but neither issue rows nor 'no results' message were definitively identified.")
                    found_issue = False

            except PlaywrightTimeoutError:
                print(f"[yellow][BrowserChecker] Timeout waiting for issue results for {repo_full_name}. Assuming no matching issues.")
                found_issue = False
            except PlaywrightError as pe:
                 print(f"[yellow][BrowserChecker] Playwright error checking results for {repo_full_name}: {pe}. Assuming no matching issues.")
                 found_issue = False


        except PlaywrightTimeoutError:
            print(f"[red][BrowserChecker] Timeout error during navigation or interaction for {repo_full_name}.")
            found_issue = False
        except PlaywrightError as pe:
             print(f"[red][BrowserChecker] Playwright error during check for {repo_full_name}: {pe}")
             found_issue = False
        except Exception as e:
            print(f"[red][BrowserChecker] Unexpected error checking {repo_full_name}: {e}")
            found_issue = False
        finally:
            if page:
                try:
                    page.close()
                except Exception as e:
                    print(f"[yellow][BrowserChecker] Error closing page: {e}")
            # Add delay regardless of success/failure to slow down requests
            print(f"[BrowserChecker] Waiting {self.check_delay:.1f}s before next check...")
            time.sleep(self.check_delay)

        return found_issue

# Example Usage (for testing)
if __name__ == "__main__":
    # repo_to_check = "octocat/Spoon-Knife" # A repo likely without 'good first issue'
    # label_to_check = "good first issue"

    repo_to_check = "microsoft/vscode" # A repo likely with 'good first issue'
    label_to_check = "good first issue"

    # repo_to_check = "facebook/react"
    # label_to_check = "good first issue" # Check a specific label

    checker = BrowserIssueChecker(headless=True, check_delay=1, timeout=20000) # Run headless for testing
    with checker: # Use context manager
        result = checker.check_repo_for_issue_label(repo_to_check, label_to_check)
        print(f"\nCheck result for {repo_to_check} with label '{label_to_check}': {result}")

        # Test another one
        # result = checker.check_repo_for_issue_label("microsoft/vscode", "help wanted")
        # print(f"\nCheck result for microsoft/vscode with label 'help wanted': {result}")
