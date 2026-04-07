<h1>Build a local RAG with Ollama</h1>

<h3>This project is based on / inspired by the following YouTube tutorial:</h3>
https://www.youtube.com/watch?v=c5jHhMXmXyo


<h2>Prerequisites</h2>
<ul>
  <li>Python 3.12.7</li>
  <li>Chat Model: llama3.1:8b</li>
  <li>Embedding Model: mxbai-embed-large</li>
</ul>



<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/LuisRodriguez3E/Local-RAG-with-Ollama-L1.git
cd ../Local-RAG-With-Ollama-L1
```


<h3>2. Ensure Ollama models are available:</h3>

```
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```


<h2>3. Run ChatBot</h2>

```
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
