<h1>Build a local RAG with Ollama</h1>

<h3>This project is based on / inspired by the following YouTube tutorial:</h3>
https://www.youtube.com/watch?v=c5jHhMXmXyo

<h2>L1 Local RAG Chatbot</h2>

```
A simple RAG application that lets users ask questions against a selected
knowledge base made from Wikipedia content.

It queries Wikipedia through the public MediaWiki API, stores the page text
locally, embeds it into Chroma, and uses a local LLM to answer questions from
that retrieved content.
```

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.12.7</li>
  <li>Ollama installed locally for embeddings</li>
  <li>Chat model available either locally through Ollama or remotely through Groq</li>
  <li>Embedding model available locally, for example <code>mxbai-embed-large</code></li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```bash
git clone https://github.com/LuisRodriguez3E/Local-RAG-with-Ollama-L1.git
cd Local-RAG-with-Ollama-L1
```

<h3>2. Create a virtual environment:</h3>

```bash
python -m venv venv
```

<h3>3. Activate the virtual environment:</h3>

```bash
venv\Scripts\activate
```

<h3>4. Install dependencies:</h3>

```bash
pip install -r requirements.txt
```

<h3>5. Ensure Ollama models are available:</h3>

```bash
ollama pull llama3.2:1b
ollama pull mxbai-embed-large
```

<h3>6. Optional: use Groq for the chat model instead of Ollama</h3>

Install the Groq LangChain integration:

```bash
pip install langchain-groq
```

Then update <code>.env</code>:

```env
EMBEDDING_MODEL = "mxbai-embed-large"
CHAT_MODEL = "llama-3.1-8b-instant"
MODEL_PROVIDER = "groq"
GROQ_API_KEY = "gsk_your_key_here"
```

The embedding model stays local in Ollama. Only the chat completion call moves to Groq.

<h2>How it works</h2>
<ul>
  <li><code>1_scraping_wikipedia.py</code> reads <code>keywords.xlsx</code> and fetches matching Wikipedia pages through the Wikipedia API.</li>
  <li><code>2_chunking_embedding_ingestion.py</code> chunks those pages and stores their embeddings in Chroma.</li>
  <li><code>3_chatbot.py</code> retrieves the most relevant chunks and answers with source links from the local vector database.</li>
</ul>

<h2>Run the app</h2>

```bash
python 1_scraping_wikipedia.py
python 2_chunking_embedding_ingestion.py
streamlit run 3_chatbot.py
```

<h3>Notes</h3>
<ul>
  <li>No Bright Data dependency is required.</li>
  <li>The Streamlit chat keeps conversation history in <code>st.session_state</code> until you clear the chat or refresh the page.</li>
  <li>If you change <code>keywords.xlsx</code>, rerun the scrape and ingestion scripts before starting the chatbot again.</li>
  <li>If you switch to Groq, make sure <code>langchain-groq</code> is installed and <code>GROQ_API_KEY</code> is set.</li>
</ul>

<h3>Chatbot</h3>
<div>
  <img src="Ragchatbot.png" alt="Screenshot of Rag chat bot" width="200"/>
</div>

<h2>Further reading</h2>
<ul>
<li>https://www.mediawiki.org/wiki/API:Main_page</li>
<li>https://python.langchain.com/docs/integrations/vectorstores/chroma/</li>
<li>https://python.langchain.com/docs/integrations/text_embedding/ollama/</li>
<li>https://ollama.com/library/mxbai-embed-large</li>
<li>https://ollama.com/library/qwen3</li>
</ul>
