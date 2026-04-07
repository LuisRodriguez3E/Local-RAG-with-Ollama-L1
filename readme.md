<h1>Build a local RAG with Ollama</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.12.7</li>
  <li>llama3.1:8b</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/LuisRodriguez3E/Local-RAG-with-Ollama-L1.git
cd Local-RAG-With-Ollama-L1
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>4. Install libraries</h3>

```
pip install -r requirements.txt
```

<h2>Executing the scripts</h2>

- Open a terminal in VS Code

- Execute the following command:

```
python run 1_scraping_wikipedia.py
python run 2_chunking_embedding_ingestion.py
streamlit run 3_chatbot.py
```

<h2>Further reading</h2>
<ul>
<li>https://www.ibm.com/think/topics/vector-embedding</li>
<li>https://ollama.com/blog/embedding-models</li>
<li>https://python.langchain.com/docs/integrations/vectorstores/chroma/</li>
<li>https://python.langchain.com/docs/integrations/text_embedding/ollama/</li>
<li>https://ollama.com/library/mxbai-embed-large</li>
<li>https://ollama.com/library/qwen3</li>
</ul>
