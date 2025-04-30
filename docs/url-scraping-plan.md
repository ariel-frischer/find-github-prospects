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
