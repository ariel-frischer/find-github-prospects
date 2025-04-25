from __future__ import annotations

import re
import time
from typing import Optional, Tuple, List  # Added Tuple, List
import urllib.parse

from playwright.sync_api import (
    Page,
    Playwright,
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)
from rich import print


class BrowserIssueChecker:
    """Uses Playwright to check GitHub frontend for open issues with a specific label."""

    # More specific selectors might be needed if GitHub UI changes
    _ISSUE_FILTER_INPUT_SELECTOR = (
        "input#repository-input"  # Updated selector based on inspection
    )
    # Selector for individual issue rows (based on provided HTML)
    _ISSUE_ROW_SELECTOR = "div.IssueRow-module__row--XmR1f"
    # Selector for the link containing the issue title and number within a row
    _ISSUE_LINK_SELECTOR = "a.IssuePullRequestTitle-module__ListItemTitle_1--_xOfg"
    # Selector for the 'no results' message container
    _NO_RESULTS_SELECTOR = "div.blankslate"
    # Regex to extract issue number from the href of the issue link
    _ISSUE_ID_RE = re.compile(r"/issues/(\d+)$")

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
    ) -> Tuple[bool, List[int]]:
        """
        Checks a repository's issues page via Playwright for open issues with the label.

        Args:
            repo_full_name: The 'owner/repo' name string.
            label: The issue label to search for.

        Returns:
            A tuple: (bool indicating if issues were found, list of found issue numbers).
        """
        if not self._browser or not self._playwright:
            print(
                "[red][BrowserChecker] Playwright/Browser not initialized. Cannot perform check."
            )
            return False

        page = None
        # Construct the raw query string, keeping the label quoted
        raw_query = f'is:issue state:open label:"{label}"'
        # URL encode the *entire* query string once using quote_plus for query parameters
        encoded_query = urllib.parse.quote_plus(raw_query)
        # Construct the final URL
        filtered_issues_url = (
            f"https://github.com/{repo_full_name}/issues?q={encoded_query}"
        )

        found_issue_flag = False
        issue_numbers: List[int] = []

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

        # Return the flag and the list of numbers (will be empty if flag is False)
        return found_issue_flag, issue_numbers


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
        has_issues, issue_nums = checker.check_repo_for_issue_label(
            repo_to_check, label_to_check
        )
        print(
            f"\nCheck result for {repo_to_check} with label '{label_to_check}': Found={has_issues}, Issues={issue_nums}"
        )

        # Test another one (e.g., a repo likely without the label)
        repo_to_check_neg = "octocat/Spoon-Knife"
        has_issues_neg, issue_nums_neg = checker.check_repo_for_issue_label(
            repo_to_check_neg, label_to_check
        )
        print(
            f"\nCheck result for {repo_to_check_neg} with label '{label_to_check}': Found={has_issues_neg}, Issues={issue_nums_neg}"
        )

        # Test another label on the first repo
        # result = checker.check_repo_for_issue_label("microsoft/vscode", "help wanted")
        # print(f"\nCheck result for microsoft/vscode with label 'help wanted': {result}")
