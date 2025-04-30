There is no single out-of-the-box API that does everything you described for free or super cheap, but you can build this workflow using lightweight, mostly free Python tools:

---

## 1. **Extract URLs from GitHub Issue Page**
- Use a tool like **URLExtract** (`urlextract` on PyPI) to extract all URLs from the raw issue text[1].
    ```python
    from urlextract import URLExtract
    extractor = URLExtract()
    urls = extractor.find_urls(issue_text)
    ```

## 2. **Scrape and Summarize Content from Each URL**
- Use **requests** and **BeautifulSoup** to fetch and parse each URL’s content[6].
- For extracting the main article/content from a page, use a library like **Goose** (`python-goose`)[8].
    ```python
    from goose3 import Goose
    g = Goose()
    article = g.extract(url=url)
    main_text = article.cleaned_text
    ```
- For summarization, you can use a free or cheap AI model (e.g., Hugging Face transformers locally, or an API like OpenAI GPT-3.5-turbo with low-cost usage, or open-source models).

## 3. **Relevance to the GitHub Issue**
- To make the summary relevant to the issue, prompt your summarizer with both the issue text and the scraped content, asking for a relevance-focused summary.

---

## Existing Tools and Automation

- **No single package** does all steps (extract URLs, fetch, extract main content, summarize with relevance to a prompt) in one go.
- **Goose** and **URLExtract** are both free and easy to use for the first two steps[1][8].
- **Summarization** can be handled by open-source models (like `t5-small`, `bart`, or `llama.cpp` locally) for free, or by cheap API calls to OpenAI, Gemini, or similar.

---

## Example Workflow (Pseudocode)
```python
from urlextract import URLExtract
from goose3 import Goose
import requests

# 1. Extract URLs
extractor = URLExtract()
urls = extractor.find_urls(issue_text)

# 2. Scrape and extract main content
g = Goose()
summaries = []
for url in urls:
    article = g.extract(url=url)
    main_text = article.cleaned_text
    # 3. Summarize with relevance (pseudo-AI call)
    summary = ai_summarize(issue_text, main_text)
    summaries.append({'url': url, 'summary': summary})
```
*Replace `ai_summarize` with your chosen summarization tool.*

Here is how to use gooose3 with litellm to get relevance per issue with each summary: 

Certainly! Here’s how you can update your workflow to use **LiteLLM** as the summarizer, ensuring summaries are relevant to the GitHub issue context.

## Incorporating LiteLLM as the Summarizer

**LiteLLM** is a Python SDK that provides a unified interface to 100+ large language models (LLMs), including OpenAI, Azure, Hugging Face, and local models. It can be used to generate summaries by calling its `completion` method with a prompt that includes both the GitHub issue text and the scraped content, ensuring relevance[1][2][5][7].

---

## Updated Example Workflow

```python
from urlextract import URLExtract
from goose3 import Goose
import litellm  # pip install litellm

# 1. Extract URLs
extractor = URLExtract()
urls = extractor.find_urls(issue_text)

# 2. Scrape and extract main content
g = Goose()
summaries = []
for url in urls:
    article = g.extract(url=url)
    main_text = article.cleaned_text

    # 3. Summarize with LiteLLM, focusing on relevance to the issue
    prompt = (
        f"We have provided context information below.\n"
        f"---------------------\n"
        f"{main_text}\n"
        f"---------------------\n"
        f"Given this information, please summarize the content with specific relevance to the following GitHub issue:\n"
        f"{issue_text}"
    )
    response = litellm.completion(
        model="gpt-3.5-turbo",  # or any supported model
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    summary = response.choices[0].message.content
    summaries.append({'url': url, 'summary': summary})

*Replace `"gpt-3.5-turbo"` with any supported model you prefer (OpenAI, Azure, Hugging Face, etc.)[1][2][7].*

---

## Key Points

- **LiteLLM** allows you to easily swap between LLM providers and models, keeping your summarization pipeline flexible and cost-effective[1][2][7].
- By constructing the prompt to include both the scraped content and the GitHub issue, you ensure the summary is focused on relevance to the issue[5].
- The rest of the pipeline (URL extraction and content scraping) remains unchanged.

---

## Summary Table (Updated)

| Step                  | Free/Cheap Tool                | Notes                                              |
|-----------------------|-------------------------------|----------------------------------------------------|
| URL Extraction        | URLExtract (`urlextract`)      | Free, simple                                       |
| Content Extraction    | Goose (`goose3`)               | Free, works on most articles                       |
| Summarization         | LiteLLM                        | Unified API, supports many models, relevance prompt|

---

**Conclusion:**  
Integrating LiteLLM as your summarizer enables you to leverage a wide range of LLMs for relevance-focused summarization in your GitHub issue pipeline, with minimal code changes and maximum flexibility[1][2][5][7].

```
---

## Summary Table

| Step                  | Free/Cheap Tool                | Notes                          |
|-----------------------|-------------------------------|-------------------------------|
| URL Extraction        | URLExtract (`urlextract`)      | Free, simple                  |
| Content Extraction    | Goose (`goose3`)               | Free, works on most articles  |
| Summarization         | HuggingFace, LLM API, etc.     | Free (local), cheap (API)     |

---

**Conclusion:**  
You can assemble a super-cheap pipeline using existing Python tools: URLExtract for URL extraction[1], Goose for content extraction[8], and open-source or cheap AI for summarization. There’s no single API that does all this out-of-the-box, but these tools together require minimal custom code and are cost-effective.

Citations:
[1] https://github.com/lipoja/URLExtract
[2] https://github.com/akash-rajak/URL-Extractor
[3] https://github.com/ryanaryap/URL-Extractor
[4] https://stackoverflow.com/questions/67503590/how-to-get-github-url-of-a-package-from-pip
[5] https://github.com/lgloege/fast-link-extractor
[6] https://brightdata.com/blog/how-tos/how-to-scrape-github-repositories-in-python
[7] https://stackoverflow.com/questions/520031/whats-the-cleanest-way-to-extract-urls-from-a-string-using-python
[8] https://github.com/grangier/python-goose

---
Answer from Perplexity: pplx.ai/share
