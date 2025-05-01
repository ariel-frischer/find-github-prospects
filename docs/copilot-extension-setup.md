# Setting Up a GitHub Copilot Extension for Issue Summarization

This guide outlines the steps to set up a GitHub Copilot Extension using an existing GitHub App. This allows building an agent or skillset that can summarize GitHub issues and their context using Copilot’s LLM API.

---

## 1. Decide: Agent or Skillset?

-   **Skillsets:** Fastest, minimal code, good for simple integrations (like summarizing issues).
-   **Agents:** More control, for complex workflows.

**For summarizing issues, a Skillset is likely fastest.**
But if you want full control, use an Agent.

---

## 2. Review Example Projects

Start with the official examples (recommended for fast setup):

-   [Skillset Example](https://github.com/copilot-extensions/skillset-example)
-   [Agent Example (Blackbeard)](https://github.com/copilot-extensions/blackbeard)

Clone one as your starting point.

---

## 3. Set Up Your Extension Locally

### a. Clone the Example Repo

```bash
git clone https://github.com/copilot-extensions/skillset-example.git
cd skillset-example
```

or for agents:

```bash
git clone https://github.com/copilot-extensions/blackbeard.git
cd blackbeard
```

### b. Install Dependencies

Most examples use Node.js or Go. For Node.js:

```bash
npm install
```

---

## 4. Configure Your GitHub App

You already have a GitHub App.
**Ensure:**

-   It has the correct permissions (read issues, pull requests, code, etc.).
-   It is registered as a Copilot Extension (see [docs](https://docs.github.com/en/copilot/building-copilot-extensions/setting-up-copilot-extensions#5-create-a-github-app-and-integrate-it-with-your-copilot-agent)).

**Update your extension’s config with:**

-   GitHub App Client ID & Secret
-   Private key
-   App ID
-   Webhook secret (if needed)

---

## 5. Connect Your App to the Extension

-   In your extension code (see the example’s README), set the environment variables or config file with your GitHub App credentials.
-   For skillsets, edit `config.json` or `.env` as needed.
-   For agents, update the agent’s config.

---

## 6. Implement Your Summarization Logic

**For Skillsets:**

-   Edit the main handler (usually `index.js` or `main.go`) to:
    -   Receive requests from Copilot Chat
    -   Fetch issue data using the GitHub API (you can use your App’s installation token)
    -   Call Copilot’s LLM API (`https://api.githubcopilot.com/chat/completions`) with the issue context
    -   Return the summary as the response

**For Agents:**

-   Implement the logic in the agent’s main handler.

---

## 7. Deploy Your Extension

-   Run locally for testing:
    ```bash
    npm start
    ```
-   Or deploy to a public server (e.g., Heroku, Vercel, AWS, etc.).

---

## 8. Register the Extension with Copilot

-   Go to [Copilot Extensions settings](https://github.com/settings/copilot/extensions) (or your org’s settings).
-   Add your extension’s public endpoint.

---

## 9. Test in Copilot Chat

-   Open Copilot Chat (in VS Code or GitHub.com)
-   Type `/extensions` to see your extension.
-   Try a prompt like:
    ```
    /your-extension summarize issue https://github.com/owner/repo/issues/123
    ```

---

## 10. (Optional) Make Public or List on Marketplace

-   Adjust visibility in your extension’s settings.

---

## Example: Summarizing an Issue (Skillset Handler Pseudocode)

```js
// Pseudocode for Node.js handler
const { Octokit } = require("@octokit/rest");
const fetch = require("node-fetch");

module.exports = async function handleRequest(req, res) {
  const issueUrl = req.body.issue_url;
  const octokit = new Octokit({ auth: process.env.GITHUB_APP_TOKEN });
  const { owner, repo, number } = parseIssueUrl(issueUrl);

  // Fetch issue and comments
  const issue = await octokit.issues.get({ owner, repo, issue_number: number });
  const comments = await octokit.issues.listComments({ owner, repo, issue_number: number });

  // Build context
  const context = buildContext(issue, comments);

  // Call Copilot LLM API
  const copilotRes = await fetch("https://api.githubcopilot.com/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.COPILOT_TOKEN}`, // Use appropriate Copilot token
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4", // Or appropriate Copilot model
      messages: [
        { role: "system", content: "You are a helpful GitHub assistant." },
        { role: "user", content: `Summarize this issue:\n${context}` }
      ]
    })
  });
  const summary = (await copilotRes.json()).choices[0].message.content;
  res.json({ summary });
};

// Helper functions (implement these)
function parseIssueUrl(url) { /* ... */ }
function buildContext(issue, comments) { /* ... */ }
```

---

## Reference Links

-   [Setting up Copilot Extensions (official docs)](https://docs.github.com/en/copilot/building-copilot-extensions/setting-up-copilot-extensions)
-   [Copilot Extensions Example Repos](https://github.com/copilot-extensions)
-   [Using Copilot’s LLM for your agent](https://docs.github.com/en/enterprise-cloud@latest/copilot/building-copilot-extensions/building-a-copilot-agent-for-your-copilot-extension/using-copilots-llm-for-your-agent)
-   [Creating a GitHub App for your Copilot Extension](https://docs.github.com/en/copilot/building-copilot-extensions/creating-a-github-app-for-your-copilot-extension)

---

## TL;DR: Fastest Path

1.  **Clone an example skillset repo**
2.  **Configure with your GitHub App credentials**
3.  **Add your summarization logic**
4.  **Deploy and register your extension**
5.  **Use it in Copilot Chat**

---

*Source: Perplexity pplx.ai/share*
