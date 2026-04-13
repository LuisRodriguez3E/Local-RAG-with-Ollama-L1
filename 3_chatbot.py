import importlib.util
import os
import re
import sys

from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage


if sys.version_info[:2] != (3, 12):
    raise RuntimeError("This project requires Python 3.12")


load_dotenv()

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}
FOLLOW_UP_TERMS = {
    "former",
    "her",
    "hers",
    "him",
    "his",
    "it",
    "its",
    "latter",
    "previous",
    "she",
    "that",
    "their",
    "them",
    "they",
    "this",
    "those",
    "these",
}


def database_exists() -> bool:
    database_location = os.getenv("DATABASE_LOCATION", "chroma_db")
    return os.path.exists(database_location)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))


@st.cache_resource(show_spinner=False)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        embedding_function=get_embeddings(),
        persist_directory=os.getenv("DATABASE_LOCATION"),
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    model_provider = os.getenv("MODEL_PROVIDER")

    if model_provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file before starting the chatbot."
            )
        if importlib.util.find_spec("langchain_groq") is None:
            raise RuntimeError(
                "The Groq integration package is not installed. Run `pip install langchain-groq`."
            )

    return init_chat_model(
        os.getenv("CHAT_MODEL"),
        model_provider=model_provider,
        temperature=0,
    )


def remove_source_block(text: str) -> str:
    cleaned_text = text.strip()
    for marker in ("\nSources:\n", "\nSource:\n", "\nSources:", "\nSource:"):
        if marker in cleaned_text:
            cleaned_text = cleaned_text.split(marker, 1)[0].strip()
    return cleaned_text


def format_chat_history(messages: list, limit: int = 6) -> str:
    recent_messages = messages[-limit:]
    if not recent_messages:
        return "No prior conversation."

    lines = []
    for message in recent_messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        content = message.content
        if isinstance(message, AIMessage):
            content = remove_source_block(content)
        lines.append(f"{role}: {content}")

    return "\n\n".join(lines)


def get_last_user_question(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content.strip()
    return ""


def extract_text(response) -> str:
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts).strip()

    return str(content).strip()


def clean_answer(answer: str) -> str:
    cleaned_answer = answer.strip()

    for marker in ("\nSources:", "\nSource:"):
        if marker in cleaned_answer:
            cleaned_answer = cleaned_answer.split(marker, 1)[0].strip()

    bad_markers = ("ToolMessage", "AIMessageChunk", "tool_calls", "invalid_tool_calls")
    if any(marker.lower() in cleaned_answer.lower() for marker in bad_markers):
        return "I don't know based on the retrieved Wikipedia data."

    return cleaned_answer or "I don't know based on the retrieved Wikipedia data."


def escape_dollar_signs(text: str) -> str:
    return text.replace("$", "\\$")


def query_tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def should_use_history(question: str) -> bool:
    lowered_question = question.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered_question) for term in FOLLOW_UP_TERMS)


def build_retrieval_context(question: str, prior_messages: list) -> tuple[bool, str, str]:
    use_history = bool(prior_messages) and should_use_history(question)
    if not use_history:
        return False, question, question

    last_user_question = get_last_user_question(prior_messages)
    if not last_user_question:
        return False, question, question

    retrieval_query = (
        f"Previous user question:\n{last_user_question}\n\n"
        f"Follow-up question:\n{question}"
    )
    scoring_question = f"{question}\nRelated topic: {last_user_question}"
    return True, retrieval_query, scoring_question


def lexical_score(question: str, document) -> float:
    tokens = query_tokens(question)
    if not tokens:
        return 0.0

    title = document.metadata.get("title", "").lower()
    content = document.page_content.lower()
    score = 0.0

    for token in tokens:
        title_matches = len(re.findall(rf"\b{re.escape(token)}\b", title))
        content_matches = len(re.findall(rf"\b{re.escape(token)}\b", content))

        score += title_matches * 8.0
        score += min(content_matches, 5) * 2.0

        if re.search(rf"^{re.escape(token)}($|[\s(])", title):
            score += 6.0

    combined_text = f"{title}\n{content}"
    if all(re.search(rf"\b{re.escape(token)}\b", combined_text) for token in tokens):
        score += 4.0

    return score


def select_relevant_documents(question: str, documents: list) -> tuple[list, list[str]]:
    scored_documents = []

    for document in documents:
        if len(document.page_content.strip()) < 40:
            continue

        score = lexical_score(question, document)
        if score <= 0:
            continue

        scored_documents.append((score, document))

    if not scored_documents:
        return [], []

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    best_score, best_document = scored_documents[0]
    primary_source = best_document.metadata.get("source", "")
    threshold = max(1.0, best_score * 0.7)

    selected_documents = []
    for score, document in scored_documents:
        if document.metadata.get("source", "") != primary_source:
            continue
        if score < threshold:
            continue

        selected_documents.append(document)
        if len(selected_documents) >= 4:
            break

    if not selected_documents:
        return [], []

    return selected_documents, [primary_source] if primary_source else []


def serialize_documents(documents: list) -> str:
    serialized_docs = []

    for index, document in enumerate(documents, start=1):
        title = document.metadata.get("title", "Untitled")
        source = document.metadata.get("source", "Unknown source")
        content = document.page_content.strip()
        serialized_docs.append(
            f"[{index}]\nTitle: {title}\nSource: {source}\nContent: {content}"
        )

    return "\n\n".join(serialized_docs)


def build_answer(question: str, prior_messages: list) -> str:
    history_text = format_chat_history(prior_messages) if prior_messages else "No prior conversation."
    use_history, retrieval_query, scoring_question = build_retrieval_context(question, prior_messages)
    history_for_prompt = history_text if use_history else "Ignore prior conversation for this question unless the user explicitly refers back to it."

    documents = get_vector_store().similarity_search(
        retrieval_query,
        k=int(os.getenv("RETRIEVAL_CANDIDATE_COUNT", "8")),
    )
    relevant_documents, sources = select_relevant_documents(scoring_question, documents)
    context = serialize_documents(relevant_documents)

    if not context:
        return "I don't know based on the retrieved Wikipedia data."

    primary_title = relevant_documents[0].metadata.get("title", "") if relevant_documents else ""

    answer_prompt = (
        "Answer only the user's current question using the retrieved Wikipedia context below.\n"
        "If the current question starts a new topic, ignore earlier conversation entirely.\n"
        "Use conversation history only when the current question clearly refers back to an earlier answer.\n"
        "Write a concise summary in one short paragraph of about 3 to 5 sentences.\n"
        "Include the most relevant details from the selected source, not just a one-line answer.\n"
        "Do not mention tools, tool messages, JSON, chunks, metadata, or internal reasoning.\n"
        "Use only the information from the selected source in the retrieved context.\n"
        "The primary topic for this answer is: " + primary_title + "\n"
        "Do not fabricate sources.\n"
        "If the context is insufficient or not relevant to the question, say exactly: "
        "I don't know based on the retrieved Wikipedia data.\n\n"
        f"Conversation history:\n{history_for_prompt}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved Wikipedia context:\n{context}\n\n"
        "Answer:"
    )

    response = get_llm().invoke(
        [
            HumanMessage(
                content=answer_prompt
            )
        ]
    )
    answer = clean_answer(extract_text(response))

    if answer == "I don't know based on the retrieved Wikipedia data.":
        return answer

    if not sources:
        return answer

    return f"{answer}\n\nSource:\n- {sources[0]}"


st.set_page_config(page_title="Wikipedia RAG Chatbot")
st.title("Wikipedia RAG Chatbot")
st.caption("Answers are generated from the ingested Wikipedia pages in your local vector database.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

if not database_exists():
    st.error(
        "Vector database not found. Run `python 1_scraping_wikipedia.py` and "
        "`python 2_chunking_embedding_ingestion.py` first."
    )
    st.stop()

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(escape_dollar_signs(message.content))

user_question = st.chat_input("Ask a question about the ingested Wikipedia pages")

if user_question:
    prior_messages = list(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the local knowledge base..."):
            try:
                ai_message = build_answer(user_question, prior_messages)
            except Exception as exc:
                ai_message = (
                    "I hit an error while answering that question. "
                    "Please verify the models are running and the vector database was "
                    f"built correctly.\n\nError: {exc}"
                )
            st.markdown(escape_dollar_signs(ai_message))

    st.session_state.messages.append(HumanMessage(user_question))
    st.session_state.messages.append(AIMessage(ai_message))
