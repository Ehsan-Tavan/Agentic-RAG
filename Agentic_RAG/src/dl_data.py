from langchain_community.document_loaders import WebBaseLoader
import json
import os

# List of URLs
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# Fetch documents
docs = [WebBaseLoader(url).load() for url in urls]

# Flatten the list of documents (since WebBaseLoader returns a list of Document objects per URL)
flattened_docs = [doc for sublist in docs for doc in sublist]

# Prepare data to save (convert Document objects to a serializable format)
docs_to_save = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in flattened_docs]

# Save to a JSON file
output_file = "../data/web_content.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(docs_to_save, f, ensure_ascii=False, indent=2)

print(f"Content saved to {output_file}")
