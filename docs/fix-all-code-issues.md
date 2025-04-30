# AI Agent Instructions: Fixing Linting and Test Issues

Your goal is to ensure the codebase is free of linting errors and passes all tests. Follow this iterative process:

1.  **Run Linting Fixes:** Execute the command `make fix`. This command uses `ruff` to automatically fix formatting and some linting issues. Review the changes made by this command.
2.  **Run Tests:** Execute the command `make test`. This command runs the `pytest` test suite.
3.  **Analyze Results:**
    *   **If `make fix` reported errors it couldn't fix:** Manually address these specific linting errors based on the `ruff` output. Only make changes necessary to satisfy the linter.
    *   **If `make test` reported failures:** Analyze the failing tests.
        *   **Prioritize fixing the test code.** Assume the application logic is correct unless proven otherwise by the test failure analysis. Modify the test assertions, setup, mocks, or logic to correctly reflect the intended behavior of the application code.
        *   **Only modify application code (`repobird_leadgen/` directory) if:**
            *   The test failure clearly indicates a bug in the application logic that contradicts its intended purpose.
            *   A linting fix required by `make fix` necessitates a change in the application code.
        *   Do **not** change application logic simply to make a poorly written test pass. Fix the test instead.
4.  **Repeat:** Go back to step 1 and repeat the cycle (`make fix`, `make test`, analyze/fix) until both `make fix` runs without making changes or reporting errors, AND `make test` runs with all tests passing (0 failures).
5.  **Final State:** The process is complete when both `make fix` and `make test` execute successfully without any errors or failures reported in their output.
