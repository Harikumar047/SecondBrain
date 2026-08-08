import streamlit as st
import os
import json
import sqlite3
import base64
from email.mime.text import MIMEText
from datetime import date, datetime, timezone
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

load_dotenv()

# ── Core setup ─────────────────────────────────────
today_str = date.today().strftime("%B %d, %Y")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key, temperature=0.4)

CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.send"]

UPLOAD_DIR = "./data/uploads"
CHROMA_DIR = "./data/chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ── Databases ──────────────────────────────────────
notes_conn = sqlite3.connect("notes.db", check_same_thread=False)
notes_conn.row_factory = sqlite3.Row
tasks_conn = sqlite3.connect("tasks.db", check_same_thread=False)
tasks_conn.row_factory = sqlite3.Row
docs_conn = sqlite3.connect("documents.db", check_same_thread=False)
docs_conn.row_factory = sqlite3.Row

notes_conn.execute("""CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL, content TEXT NOT NULL,
    tags TEXT DEFAULT '', created_at TEXT NOT NULL)""")

tasks_conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL, status TEXT DEFAULT 'pending',
    due_date TEXT DEFAULT '', created_at TEXT NOT NULL,
    completed_at TEXT DEFAULT '')""")

docs_conn.execute("""CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    title TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    uploaded_at TEXT NOT NULL)""")

notes_conn.commit()
tasks_conn.commit()
docs_conn.commit()

def now():
    return datetime.now(timezone.utc).isoformat()

# ── Long-term memory ───────────────────────────────
long_term_store = {}

def save_to_memory(key: str, value: str, user_id: str = "user1"):
    if user_id not in long_term_store:
        long_term_store[user_id] = {}
    long_term_store[user_id][key] = value

def get_from_memory(key: str, user_id: str = "user1"):
    return long_term_store.get(user_id, {}).get(key)

def get_all_memories(user_id: str = "user1"):
    return long_term_store.get(user_id, {})

# ── Gmail ──────────────────────────────────────────
def get_gmail_service():
    creds = None
    if "GMAIL_TOKEN_JSON" in st.secrets:
        try:
            token_info = json.loads(st.secrets["GMAIL_TOKEN_JSON"])
            creds = Credentials.from_authorized_user_info(token_info, SCOPES)
        except Exception as e:
            st.error(f"Failed to load GMAIL_TOKEN_JSON from secrets: {e}")
    elif os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception:
            pass

    if creds:
        try:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
            return build("gmail", "v1", credentials=creds)
        except Exception as e:
            st.error(f"Gmail initialization failed: {e}")
            return None
    return None

gmail_service = get_gmail_service()

# ── Embeddings / Vectorstore / Splitter (shared) ───
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

# ── Document ingestion (Module 2, dynamic) ─────────
def derive_title(filename: str) -> str:
    """Turn 'project_alpha_design.pdf' into 'Project Alpha Design'."""
    name = os.path.splitext(os.path.basename(filename))[0]
    return name.replace("_", " ").replace("-", " ").title()

def load_document(filepath: str):
    ext = filepath.lower().split(".")[-1]
    if ext == "pdf":
        loader = PyPDFLoader(filepath)
    elif ext == "txt":
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")
    return loader.load()

def ingest_document(filepath: str) -> str:
    """Load, chunk, embed, and register an uploaded document. Called dynamically
    the moment a file is uploaded through the UI — no restart needed."""
    filename = os.path.basename(filepath)
    title = derive_title(filename)
    ext = filepath.lower().split(".")[-1]

    raw_docs = load_document(filepath)
    chunks = splitter.split_documents(raw_docs)

    for c in chunks:
        c.metadata["source"] = filename
        c.metadata["title"] = title

    vectorstore.add_documents(chunks)

    docs_conn.execute(
        "INSERT INTO documents (filename, title, doc_type, uploaded_at) VALUES (?, ?, ?, ?)",
        (filename, title, ext, now())
    )
    docs_conn.commit()

    return f"Ingested '{title}' ({len(chunks)} chunks) as {filename}."

def find_document(name_query: str) -> list:
    """Fuzzy-match a spoken document name against the registry."""
    words = [w.strip().lower() for w in name_query.split() if len(w.strip()) > 2]
    if not words:
        return []
    rows = docs_conn.execute("SELECT filename, title FROM documents").fetchall()
    matches = []
    for r in rows:
        haystack = f"{r['filename']} {r['title']}".lower()
        if any(w in haystack for w in words):
            matches.append(r["filename"])
    return matches

# ── Tools ──────────────────────────────────────────
@tool
def add_note(title: str, content: str, tags: str = "") -> str:
    """Save a new note with title, content, and optional tags."""
    cur = notes_conn.execute(
        "INSERT INTO notes (title, content, tags, created_at) VALUES (?, ?, ?, ?)",
        (title, content, tags, now()))
    notes_conn.commit()
    return f"Note saved: '{title}'."

@tool
def list_notes(limit: int = 20) -> str:
    """List recent notes."""
    rows = notes_conn.execute(
        "SELECT id, title, tags FROM notes ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    if not rows:
        return "No notes found."
    return "\n".join(f"[{r['id']}] {r['title']} (tags: {r['tags']})" for r in rows)

@tool
def search_notes(query: str) -> str:
    """Search notes by keyword."""
    words = [w.strip().lower() for w in query.split() if len(w.strip()) > 2]
    if not words:
        return "Please provide a more specific search term."
    variants = set()
    for w in words:
        variants.add(w)
        variants.add(w[:-1] if w.endswith("s") else w + "s")
    conditions = " OR ".join(
        ["(LOWER(title) LIKE ? OR LOWER(content) LIKE ? OR LOWER(tags) LIKE ?)"] * len(variants))
    params = []
    for v in variants:
        like = f"%{v}%"
        params.extend([like, like, like])
    rows = notes_conn.execute(
        f"SELECT id, title, content FROM notes WHERE {conditions}", params).fetchall()
    if not rows:
        return f"No notes matched '{query}'."
    return "\n\n".join(f"[{r['id']}] {r['title']}\n{r['content']}" for r in rows)

@tool
def add_task(description: str, due_date: str = "") -> str:
    """Add a new task with optional due date (YYYY-MM-DD)."""
    cur = tasks_conn.execute(
        "INSERT INTO tasks (description, due_date, created_at) VALUES (?, ?, ?)",
        (description, due_date, now()))
    tasks_conn.commit()
    return f"Task added: '{description}'."

@tool
def list_pending_tasks() -> str:
    """List all pending tasks."""
    rows = tasks_conn.execute(
        "SELECT id, description, due_date FROM tasks WHERE status = 'pending' ORDER BY due_date"
    ).fetchall()
    if not rows:
        return "No pending tasks."
    return "\n".join(f"[{r['id']}] {r['description']} (due: {r['due_date'] or 'n/a'})" for r in rows)

@tool
def complete_task(task_id: int) -> str:
    """Mark a task as complete by id."""
    tasks_conn.execute(
        "UPDATE tasks SET status = 'complete', completed_at = ? WHERE id = ?",
        (now(), task_id))
    tasks_conn.commit()
    return f"Task {task_id} marked complete."

@tool
def read_inbox(max_results: int = 5) -> str:
    """Read latest emails from Gmail inbox."""
    if not gmail_service:
        return "Gmail integration is not configured or authenticated. Please verify credentials."
    results = gmail_service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["INBOX"]
    ).execute()
    messages = results.get("messages", [])
    if not messages:
        return "No emails found."
    output = []
    for msg in messages:
        txt = gmail_service.users().messages().get(userId="me", id=msg["id"]).execute()
        headers = txt["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
        snippet = txt.get("snippet", "")[:150]
        output.append(f"From: {sender}\nSubject: {subject}\nPreview: {snippet}")
    return "\n\n".join(output)

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail."""
    if not gmail_service:
        return "Gmail integration is not configured or authenticated. Please verify credentials."
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"Email sent to {to} with subject '{subject}'."

search_tool = DuckDuckGoSearchRun(description="Search the web for current information.")
wiki_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)

@tool
def knowledge_base_search(query: str) -> str:
    """Answer a specific question using uploaded documents. Automatically narrows to a
    named document (e.g. 'Project Alpha') if the query mentions one."""
    matched_files = find_document(query)
    search_kwargs = {"k": 5}
    if matched_files:
        search_kwargs["filter"] = {"source": {"$in": matched_files}}

    results = vectorstore.similarity_search(query, **search_kwargs)
    if not results:
        return "I don't have that information in the uploaded documents."

    context = "\n\n".join(f"[{r.metadata.get('title')}] {r.page_content}" for r in results)
    prompt = f"""Answer using ONLY the context below. If the answer isn't in the context, say:
"I don't have that information in the uploaded documents."

Context:
{context}

Question:
{query}"""
    return llm.invoke(prompt).content

@tool
def summarize_document(document_name: str) -> str:
    """Summarize an entire uploaded document by name (e.g. 'Meeting Minutes from Monday')."""
    matched_files = find_document(document_name)
    if not matched_files:
        return f"No uploaded document matches '{document_name}'."

    all_chunks = vectorstore.get(where={"source": {"$in": matched_files}})
    docs_and_meta = list(zip(all_chunks["documents"], all_chunks["metadatas"]))
    docs_and_meta.sort(key=lambda x: x[1].get("page", 0))

    full_text = "\n".join(d for d, _ in docs_and_meta)
    if not full_text.strip():
        return f"Document '{document_name}' has no content indexed."

    prompt = f"""Summarize the following document clearly and concisely, covering the
main points, decisions, and any action items if present.

Document:
{full_text[:12000]}"""
    return llm.invoke(prompt).content

@tool
def list_documents() -> str:
    """List all uploaded documents available for search."""
    rows = docs_conn.execute(
        "SELECT title, doc_type FROM documents ORDER BY uploaded_at DESC").fetchall()
    if not rows:
        return "No documents uploaded yet."
    return "\n".join(f"• {r['title']} ({r['doc_type']})" for r in rows)

# ── Agents ─────────────────────────────────────────
notes_agent = create_agent(llm,
    tools=[add_note, list_notes, search_notes],
    system_prompt=f"""You are a notes management assistant. Today: {today_str}.
Use tools AT MOST ONCE per request. Give clear confirmations.""")

tasks_agent = create_agent(llm,
    tools=[add_task, list_pending_tasks, complete_task],
    system_prompt=f"""You are a task management assistant. Today: {today_str}.
Use tools AT MOST ONCE per request. Give clear confirmations.""")

research_agent = create_agent(llm,
    tools=[search_tool, wikipedia_tool],
    system_prompt=f"""You are a research assistant. Today: {today_str}.
Use each tool AT MOST ONCE. Never repeat queries with reworded searches.""")

email_agent = create_agent(llm,
    tools=[read_inbox, send_email],
    system_prompt=f"""You are an email assistant. Today: {today_str}.
Use tools AT MOST ONCE per request. Summarize emails clearly.""")

knowledge_agent = create_agent(llm,
    tools=[knowledge_base_search, summarize_document, list_documents],
    system_prompt=f"""You are a knowledge base assistant. Today: {today_str}.
You have access to uploaded PDFs, TXT files, study notes, meeting minutes, and manuals.

Rules:
- Use knowledge_base_search for specific questions.
- Use summarize_document when asked to summarize a named document.
- Use list_documents if the user asks what's available.
- Call each tool AT MOST ONCE per request.
- Only answer from document content — never invent facts.""")

# ── Intent Classifier ──────────────────────────────
class IntentClassifier(BaseModel):
    intent: Literal["notes", "tasks", "research", "knowledge", "email", "chat"]
    reasoning: str

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify the user message into exactly one intent:
- notes: saving, listing, searching, updating notes
- tasks: adding, listing, completing tasks
- research: searching web or Wikipedia for information
- knowledge: querying, searching, or summarizing uploaded documents
- email: reading or sending emails
- chat: general conversation, greetings, personal info, anything else
Always return a structured classification. Never respond in plain text."""),
    ("human", "{input}")
])
classifier = classifier_prompt | llm.with_structured_output(IntentClassifier)

# ── Coordinator ────────────────────────────────────
def run_coordinator(user_input: str, thread_id: str = "default") -> str:
    classification = classifier.invoke({"input": user_input})
    intent = classification.intent

    user_lower = user_input.lower()
    memory_patterns = {
        "my name is": "name",
        "call me": "name",
        "i am": "description",
        "i like": "preference",
        "i prefer": "preference",
        "i work at": "workplace",
        "i work in": "field",
        "my email is": "email",
        "my job is": "job",
    }
    for pattern, key in memory_patterns.items():
        if pattern in user_lower:
            value = user_input[user_lower.index(pattern) + len(pattern):].strip()
            if value:
                save_to_memory(key, value)
            break

    agent_map = {
        "notes": notes_agent,
        "tasks": tasks_agent,
        "research": research_agent,
        "email": email_agent,
        "knowledge": knowledge_agent,
    }

    if intent == "chat":
        memories = get_all_memories()
        memory_context = "\n".join(
            f"{k}: {v}" for k, v in memories.items()) if memories else "No stored info yet."
        response = llm.invoke(f"""You are Second Brain, a helpful personal productivity assistant.

Known facts about the user:
{memory_context}

User message: {user_input}

Respond naturally and helpfully. Use the known facts where relevant.""")
        return response.content

    agent = agent_map[intent]
    events = agent.stream(
        {"messages": [("user", user_input)]},
        config={"recursion_limit": 8, "configurable": {"thread_id": thread_id}},
        stream_mode="values"
    )
    final = None
    for event in events:
        final = event["messages"][-1]
    return final.content

# ── Daily Briefing ─────────────────────────────────
def generate_briefing():
    pending_tasks = [row[0] for row in tasks_conn.execute(
        "SELECT description FROM tasks WHERE status = 'pending'").fetchall()]
    recent_notes = [row[0] for row in notes_conn.execute(
        "SELECT title FROM notes ORDER BY created_at DESC LIMIT 5").fetchall()]
    memories = get_all_memories()

    class DailyBriefing(BaseModel):
        date: str = Field(description="Today's date")
        summary: str = Field(description="2-3 sentence overview of the day")
        pending_tasks: List[str] = Field(description="List of pending tasks")
        recent_notes: List[str] = Field(description="List of recent note titles")
        focus_suggestion: str = Field(description="One thing to focus on today")
        motivational_note: str = Field(description="A short motivational sentence")

    briefing_llm = llm.with_structured_output(DailyBriefing)
    user_context = "\n".join(f"{k}: {v}" for k, v in memories.items()) if memories else ""
    prompt = f"""Generate a daily briefing for today: {today_str}
User info: {user_context}
Pending tasks: {pending_tasks}
Recent notes: {recent_notes}
Fill all fields of the briefing schema."""
    return briefing_llm.invoke(prompt)

# ── Streamlit UI ───────────────────────────────────
st.set_page_config(page_title="Second Brain", page_icon="🧠", layout="wide")
st.title("🧠 Second Brain — AI Productivity Assistant")
st.caption(f"Today: {today_str}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_1"

# ── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.header("⚡ Quick Actions")

    if gmail_service:
        st.success("🟢 Gmail Connected")
    else:
        st.error("🔴 Gmail Disconnected")
        available_keys = list(st.secrets.keys()) if st.secrets else []
        st.info(f"To connect, add your `token.json` content to Streamlit Secrets as `GMAIL_TOKEN_JSON`. Available keys: {available_keys}")

    if st.button("📅 Daily Briefing", use_container_width=True):
        with st.spinner("Generating your briefing..."):
            try:
                briefing = generate_briefing()
                st.success("Briefing ready!")
                st.subheader(f"📅 {briefing.date}")
                st.write(f"**{briefing.summary}**")
                if briefing.pending_tasks:
                    st.write("**✅ Tasks:**")
                    for t in briefing.pending_tasks:
                        st.write(f"• {t}")
                if briefing.recent_notes:
                    st.write("**📝 Notes:**")
                    for n in briefing.recent_notes:
                        st.write(f"• {n}")
                st.info(f"🎯 **Focus:** {briefing.focus_suggestion}")
                st.success(f"💪 {briefing.motivational_note}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("📧 Check Inbox", use_container_width=True):
        with st.spinner("Reading emails..."):
            try:
                result = run_coordinator("Show me my latest 5 emails", st.session_state.thread_id)
                st.write(result)
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # ── Dynamic document upload ─────────────────────
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        key="doc_uploader"
    )
    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.caption(f"Ready to ingest: {uploaded_file.name}")

        if st.button("⚙️ Ingest this document", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    result = ingest_document(save_path)
                    st.success(result)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to ingest: {e}")

    st.subheader("📚 Uploaded Documents")
    doc_rows = docs_conn.execute(
        "SELECT title, doc_type FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    if doc_rows:
        for r in doc_rows:
            st.write(f"• {r['title']} ({r['doc_type']})")
    else:
        st.write("No documents uploaded yet.")

    st.divider()

    st.subheader("📝 Recent Notes")
    rows = notes_conn.execute(
        "SELECT title FROM notes ORDER BY created_at DESC LIMIT 5").fetchall()
    if rows:
        for r in rows:
            st.write(f"• {r['title']}")
    else:
        st.write("No notes yet.")

    st.divider()

    st.subheader("✅ Pending Tasks")
    rows = tasks_conn.execute(
        "SELECT description, due_date FROM tasks WHERE status = 'pending'"
    ).fetchall()
    if rows:
        for r in rows:
            st.write(f"• {r['description']} (due: {r['due_date'] or 'n/a'})")
    else:
        st.write("No pending tasks.")

    st.divider()

    st.subheader("🧠 Memory")
    memories = get_all_memories()
    if memories:
        for k, v in memories.items():
            st.write(f"**{k}:** {v}")
    else:
        st.write("No memories stored yet.")

st.subheader("💬 Chat")
st.caption("Ask anything — notes, tasks, emails, research, documents, or just chat!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = run_coordinator(prompt, st.session_state.thread_id)
            except Exception as e:
                response = f"Sorry, I encountered an error: {e}"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})