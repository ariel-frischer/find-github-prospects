# RepoBird LeadGen: GitHub Repo & Issue Analysis Tool

This repository contains the `repobird-leadgen` tool, a Python CLI application designed to:
1.  Discover relevant GitHub repositories based on search criteria.
2.  Identify specific open issues within those repositories.
3.  Scrape the content of those issues (title, body, comments).
4.  Analyze the scraped issue content using an LLM (via LiteLLM) to assess complexity, suitability for AI agents, and other factors.

## RepoBird LeadGen Tool

`repobird-leadgen` helps find GitHub repositories matching criteria (issue labels, language, stars, activity), identifies relevant issues, scrapes their content, performs LLM-based analysis, and saves the structured results.

### Installation

1.  Create and activate a virtual environment using `uv` (assuming you are in the project root):
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
2.  Install the tool and its dependencies (including development tools):
    ```bash
    uv pip install -e .[dev]
    ```
    *Note: Using `-e .[dev]` installs the package in editable mode along with development dependencies (like `pytest`, `pytest-mock`, `ruff`, `pydantic`) specified in `pyproject.toml`.*
3.  **Install Playwright Browsers:** The issue scraping functionality relies on Playwright. Install the necessary browser binaries:
    ```bash
    playwright install --with-deps
    ```
    *Note: `--with-deps` attempts to install needed OS dependencies. This might require `sudo` and may not work on all systems. If it fails, you might need to install browser dependencies manually based on Playwright documentation for your OS.*

### Configuration

`repobird-leadgen` requires API keys for GitHub and the LLM provider (via LiteLLM).

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Edit the newly created `.env` file and add your GitHub PAT:
    ```dotenv
    # .env
    # .env
    GITHUB_TOKEN="your_github_pat_here"

    # LiteLLM Configuration (Example for OpenRouter)
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    # Optional: Specify the LLM model if you don't want the default
    # LLM_MODEL="openrouter/google/gemini-pro-1.5"
    ```
    *See the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for environment variables required by other providers (OpenAI, Anthropic, Azure, etc.).*

You can also optionally configure the following settings in the `.env` file:
*   `CONCURRENCY=20` (Number of parallel workers for issue scraping/analysis, default: 20)
*   `CACHE_DIR=cache` (Directory to store raw search results cache, default: "cache")
*   `OUTPUT_DIR=output` (Directory to save final enriched analysis files, default: "output")
*   `LLM_MODEL="openrouter/google/gemini-flash-1.5"` (Default LLM model used for analysis)

### Usage

After installation, run the tool from within the `repobird-leadgen` directory using the `repobird-leadgen` command.

The tool has four main subcommands:

**1. `search`**

Finds repositories matching your criteria and saves the raw data to a JSON cache file.

*   **Usage:**
    ```bash
    repobird-leadgen search [OPTIONS]
    ```
*   **Options:**
    *   `--label` / `-l` TEXT: Issue label to search for (default: "good first issue").
    *   `--language` / `-lang` TEXT: Repository programming language (default: "python").
    *   `--max-results` / `-n` INTEGER: Maximum number of repositories to fetch (default: 30).
    *   `--min-stars` / `-s` INTEGER: Minimum number of stars a repository must have (default: 20).
    *   `--recent-days` / `-d` INTEGER: Maximum number of days since the last push activity (default: 365).
    *   `--cache-file` / `-c` PATH: Path to save the raw repository data as a JSON Lines (`.jsonl`) file.
        *   **How it works:** The `search` command saves the raw JSON data for each repository that meets *all* specified criteria (including label, age, PR filters) to this file, one JSON object per line.
        *   **Incremental & Skipping:** If the specified cache file already exists, the command first reads it to identify repositories that have already been processed and saved. It then skips checking these repositories again during the current search. Any *new* qualifying repositories found are appended to the existing file. This prevents duplicate entries and saves API calls on subsequent runs with the same cache file.
        *   **Default Naming:** If `--cache-file` is not provided, a filename is automatically generated based on the search parameters (label, language, stars, days, age, prs) and placed in the `cache/` directory (e.g., `cache/raw_repos_label_good_first_issue_lang_python_stars_20_days_365_age_30_prs_0.jsonl`).
    *   `--max-issue-age-days` INTEGER: [Filter] Only find repos where at least one matching issue was created within this many days (optional).
    *   `--max-linked-prs` INTEGER: [Filter] Only find repos where at least one matching issue has this many or fewer linked pull requests (optional). Uses API checks by default.
    *   `--use-browser-checker`: [Filter] Use Playwright browser automation for certain checks. The behavior depends on whether detailed filters (`--max-issue-age-days`, `--max-linked-prs`) are active:
        *   **No Detailed Filters:** If neither `--max-issue-age-days` nor `--max-linked-prs` is specified, using `--use-browser-checker` makes the tool perform the *initial* check for issues matching the `--label` by scraping the GitHub issues page directly, instead of using the GitHub API's `search_issues`. This can sometimes find issues the API might miss but is slower.
        *   **Detailed Filters Active (Hybrid Approach):** When `--max-issue-age-days` and/or `--max-linked-prs` are specified along with `--use-browser-checker`:
            1.  The GitHub API is *always* used for the initial repository search (finding repos matching language, stars, activity).
            2.  The API is *always* used to find *candidate* issues within those repos that match the `--label` and the `--max-issue-age-days` filter.
            3.  **If `--max-linked-prs 0` is specified:** The browser checker then takes the list of candidate issues (that passed the age filter) and verifies which ones *actually* have zero linked PRs by performing a specific search query directly on the GitHub issues UI (`.../issues?q=... -linked:pr ...`). This UI check is often more accurate for the zero-PR case than the API's timeline events. Only issues confirmed by the browser to have zero PRs are considered qualified.
            4.  **If `--max-linked-prs N` (where N > 0) is specified:** The browser checker is *not* currently used for the PR check. The API's timeline event check is used to count linked PRs for the candidate issues (similar to the default behavior without `--use-browser-checker`).
*   **Example with detailed filters (API default):** Find Python repos with "good first issue" issues created in the last 30 days that have 0 linked PRs, saving to a specific cache file:
    ```bash
    repobird-leadgen search --label "good first issue" --language python --max-issue-age-days 30 --max-linked-prs 0 --cache-file cache/my_search_results.jsonl
    ```
*   **Example with hybrid browser check for zero PRs:** Find Python repos with "good first issue" issues created in the last 30 days, using the browser to verify which ones have exactly 0 linked PRs:
    ```bash
    repobird-leadgen search --label "good first issue" --language python --max-issue-age-days 30 --max-linked-prs 0 --use-browser-checker --cache-file cache/my_search_results_hybrid.jsonl
    ```

**2. `enrich`**

Reads a cache file generated by the `search` command, scrapes the content (title, body, comments) of each identified issue using Playwright, analyzes the content using an LLM (via LiteLLM), and saves the structured analysis results along with the original repository data to a new JSON Lines file.

*   **Usage:**
    ```bash
    repobird-leadgen enrich <CACHE_FILE> [OPTIONS]
    ```
*   **Argument:**
    *   `CACHE_FILE`: Path to the JSON Lines (`.jsonl`) cache file created by the `search` command (contains repo info including `found_issues`).
*   **Options:**
    *   `--output-dir` / `-o` PATH: Directory to save the final enriched JSON Lines file (default: "output"). The output filename will be based on the input cache file name (e.g., `enriched_my_search_results.jsonl`).
    *   `--concurrency` / `-w` INTEGER: Number of parallel workers for scraping and analyzing issues (default: 20).
*   **Example:** Enrich the results from the previous search example:
    ```bash
    repobird-leadgen enrich cache/my_search_results.jsonl --output-dir output/analysis --concurrency 10
    ```
    *   **How it works:** The `enrich` command reads the specified cache file line by line. For each repository entry, it identifies the issues listed in the `found_issues` field. It then uses Playwright (a browser automation tool) to visit each issue's URL on GitHub, scraping the title, body, and comments (currently only those initially loaded, not handling pagination). The scraped text content for each issue is then formatted into a prompt and sent to the configured LLM (via LiteLLM) along with a JSON schema (`LLMIssueAnalysisSchema`). The LLM analyzes the issue and (ideally) returns a JSON object matching the schema, containing fields like `full_problem_statement`, `complexity_score`, `is_good_first_issue_for_agent`, etc. This analysis result is combined with the original repository data.
    *   **Parallel Processing & Incremental Saving:** To handle potentially many repositories and issues efficiently and robustly, the `enrich` command processes repositories in parallel using multiple worker processes (up to the `--concurrency` limit). Crucially, as soon as a worker finishes processing *all* issues for a single repository, the resulting enriched data (original repo info + list of issue analyses) is immediately written as a single JSON line to the final output file. This incremental saving ensures that if the process crashes or is interrupted, the results for already completed repositories are preserved, preventing data loss and wasted computation/API calls. A dedicated writer process manages safe, sequential writes to the output file from the parallel workers.

**3. `post_process`**

Filters an enriched JSONL file (output from `enrich`) to create a new file containing only repositories that have at least one issue marked as suitable for an AI agent.

*   **Usage:**
    ```bash
    repobird-leadgen post_process <ENRICHED_FILE>
    ```
*   **Argument:**
    *   `ENRICHED_FILE`: Path to the JSON Lines (`.jsonl`) file created by the `enrich` command.
*   **Output:** Creates a new file named `filtered_<original_name>.jsonl` in the same directory as the input file. This file contains only the lines (repositories) where at least one entry in the `issue_analysis` list has `"is_good_first_issue_for_agent": true`.
*   **Example:** Filter the results from the previous enrichment example:
    ```bash
    repobird-leadgen post_process output/analysis/enriched_my_search_results.jsonl
    ```
    This would create `output/analysis/filtered_enriched_my_search_results.jsonl`.

**4. `full`**

Runs the `search` and `enrich` steps sequentially. It uses a temporary cache file between the steps unless `--keep-cache` is specified. Outputs the final enriched JSON Lines file.

*   **Usage:**
    ```bash
    repobird-leadgen full [OPTIONS]
    ```
*   **Options:** (Combines options from `search` and `enrich`, excluding `--cache-file` from `search`)
    *   `--label` / `-l` TEXT (default: "good first issue")
    *   `--language` / `-lang` TEXT (default: "python")
    *   `--max-results` / `-n` INTEGER (default: 30)
    *   `--min-stars` / `-s` INTEGER (default: 20)
    *   `--recent-days` / `-d` INTEGER (default: 365) [Search Phase]
    *   `--max-issue-age-days` INTEGER (optional) [Search Phase]
    *   `--max-linked-prs` INTEGER (optional) [Search Phase]
    *   `--use-browser-checker`: [Search Phase] Use browser automation (see `search` help for details on hybrid approach).
    *   `--output-dir` / `-o` PATH (default: "output") [Enrich Phase] Directory for final enriched JSONL file.
    *   `--concurrency` / `-w` INTEGER (default: 20) [Enrich Phase] Parallel workers for scraping/analysis.
    *   `--keep-cache`: Keep the intermediate raw repository cache file from the search step.
*   **Example:** Run the full pipeline for Python repos with "help wanted" issues, keeping the intermediate cache:
    ```bash
    repobird-leadgen full --label "help wanted" --language python --min-stars 50 --keep-cache --output-dir output/full_run_analysis
    ```

## Roadmap / Future Enhancements

*   **URL Content Extraction & Summarization:**
    *   Identify URLs mentioned within issue bodies and comments.
    *   For relevant URLs (e.g., documentation, related code snippets, external discussions), scrape their content.
    *   Generate concise summaries of the scraped URL content.
    *   Incorporate these summaries into the prompt sent to the LLM during the `enrich` phase to provide richer context for the `IssueAnalysis`. This could improve the accuracy of complexity scores and suitability assessments.
*   **Advanced Comment Scraping:** Implement logic to handle pagination or dynamic loading of comments on GitHub issue pages to capture the full discussion context for very long threads.
*   **Configurable LLM Prompts:** Allow users to customize the system prompt used for issue analysis via configuration files.
*   **Alternative Output Formats:** Reintroduce support for different output formats (e.g., CSV, Markdown table) for the final enriched data, potentially using libraries like `pandas` for easier conversion.
*   **Copilot Extension Integration:** Explore creating a GitHub Copilot Extension (Skillset or Agent) to leverage the Copilot LLM API (`https://api.githubcopilot.com/chat/completions`) for generating more context-aware issue summaries. This could involve fetching issue data, comments, and linked issues via the GitHub API within the extension and using the Copilot LLM for pre-summarization before the main `enrich` step. See [docs/copilot-extension-setup.md](docs/copilot-extension-setup.md) for setup details.
*   **Copilot Extension Integration:** Explore creating a GitHub Copilot Extension (Skillset or Agent) to leverage the Copilot LLM API (`https://api.githubcopilot.com/chat/completions`) for generating more context-aware issue summaries. This could involve fetching issue data, comments, and linked issues via the GitHub API within the extension and using the Copilot LLM for pre-summarization before the main `enrich` step. See [docs/copilot-extension-setup.md](docs/copilot-extension-setup.md) for setup details.
