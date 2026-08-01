import streamlit as st
import ollama
import json
import re
import pandas as pd
import sqlite3

from dataclasses import asdict
from datetime import datetime
from ddgs import DDGS
from pypdf import PdfReader
from typing import TypedDict, List, Dict, Callable
from langgraph.graph import StateGraph, START, END
from video_agent import (
    DEFAULT_ASPECT_RATIO_LABEL,
    DEFAULT_FRAME_RATE,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_LANGUAGE,
    DEFAULT_MOTION_LEVEL,
    DEFAULT_NARRATION_ENABLED,
    DEFAULT_SPEAKING_SPEED,
    DEFAULT_SPEAKING_STYLE,
    DEFAULT_TOTAL_DURATION_SECONDS,
    DEFAULT_SCENE_DURATION_SECONDS,
    DEFAULT_VISUAL_STYLE,
    DEFAULT_VOICE,
    VideoPlan,
    build_video_plan,
    build_video_settings,
    settings_snapshot,
)
from video_providers import generate_narration_audio, generate_scene_images
from video_renderer import build_generation_output_dir, ensure_ffmpeg_available, render_video


# -----------------------------
# App State
# Comes from: Python TypedDict
# Role: defines the data LangGraph passes between nodes
# -----------------------------
class AumState(TypedDict):
    question: str
    route: str
    route_reason: str
    route_confidence: float
    answer: str
    messages: List[Dict[str, str]]
    tool_result: str
    csv_summary: str
    pdf_text: str
    pdf_intelligence: str
    pdf_chunks: List[Dict[str, str]]
    web_sources: List[Dict[str, str]]
    video_request: str
    video_content_type: str
    video_plan_json: str
    video_output_dir: str
    video_image_paths: List[str]
    video_audio_path: str
    video_file_path: str
    video_status_messages: List[str]


# -----------------------------
# Persistent Memory: SQLite
# Comes from: Python sqlite3
# Role: save/load chat history and long-term user facts
# -----------------------------
DB_PATH = "aumstate_memory.db"


def init_memory_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_message_to_db(role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_memory (role, content, created_at)
        VALUES (?, ?, ?)
        """,
        (role, content, datetime.now().isoformat())
    )

    conn.commit()
    conn.close()


def save_user_fact(fact_key: str, fact_value: str):
    """
    Simple upsert behavior.

    Instead of adding duplicate facts like:
    - name: Leelu
    - name: Leelu

    This deletes the old value for the same fact_key and inserts the latest value.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM user_facts WHERE fact_key = ?",
        (fact_key,)
    )

    cursor.execute(
        """
        INSERT INTO user_facts (fact_key, fact_value, created_at)
        VALUES (?, ?, ?)
        """,
        (fact_key, fact_value, datetime.now().isoformat())
    )

    conn.commit()
    conn.close()


def load_user_facts(limit: int = 50) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT fact_key, fact_value
        FROM user_facts
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No saved user facts yet."

    rows = rows[::-1]

    facts = []
    for key, value in rows:
        facts.append(f"- {key}: {value}")

    return "\n".join(facts)


def clear_user_facts():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM user_facts")

    conn.commit()
    conn.close()


def load_messages_from_db(limit: int = 30):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM chat_memory
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    rows = rows[::-1]

    return [
        {"role": role, "content": content}
        for role, content in rows
    ]


def clear_memory_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chat_memory")

    conn.commit()
    conn.close()


# -----------------------------
# User Fact Extractor
# Comes from: Python regex/rules
# Role: save important facts separately from raw chat history
# -----------------------------
def extract_and_save_user_facts(user_text: str):
    text = user_text.strip()
    lower = text.lower()

    name_patterns = [
        r"my name is ([a-zA-Z]+)",
        r"i am ([a-zA-Z]+)",
        r"i'm ([a-zA-Z]+)"
    ]

    for pattern in name_patterns:
        match = re.search(pattern, lower)
        if match:
            name = match.group(1).capitalize()
            save_user_fact("name", name)

    if "aum state" in lower or "aumstate" in lower:
        save_user_fact("project", "User is building AUM State.")

    if "practical" in lower and "code" in lower:
        save_user_fact("preference", "User prefers practical code-focused explanations.")


# -----------------------------
# Memory helper
# Role: updates Streamlit memory and SQLite memory together
# -----------------------------
def update_memory(state: AumState, answer: str):
    user_message = {"role": "user", "content": state["question"]}
    assistant_message = {"role": "assistant", "content": answer}

    save_message_to_db("user", state["question"])
    save_message_to_db("assistant", answer)

    extract_and_save_user_facts(state["question"])

    return state["messages"] + [
        user_message,
        assistant_message
    ]


# -----------------------------
# Deterministic Route Override
# Comes from: Python rules
# Role: catches obvious tool cases before asking the LLM router
# -----------------------------
def deterministic_route_override(state: AumState):
    question = state["question"].lower()

    has_pdf = bool(state.get("pdf_text", "").strip())
    has_csv = bool(state.get("csv_summary", "").strip())

    pdf_reference_words = [
        "this book",
        "this pdf",
        "this document",
        "this file",
        "this report",
        "this chapter",
        "the book",
        "the pdf",
        "the document",
        "the file",
        "summarize this",
        "explain this",
        "interesting facts",
        "key points",
        "main points",
        "what is this about"
    ]

    csv_reference_words = [
        "this csv",
        "this data",
        "this dataset",
        "uploaded csv",
        "uploaded data",
        "analyze this data",
        "analyze this csv",
        "rows",
        "columns"
    ]

    web_reference_words = [
        "search web",
        "search the web",
        "browse",
        "internet",
        "online",
        "latest",
        "current news",
        "today's news",
        "today",
        "recently",
        "current version",
        "current docs"
    ]

    if has_pdf and any(word in question for word in pdf_reference_words):
        return {
            "route": "pdf_reader",
            "route_reason": "A PDF is uploaded and the user is referring to the current book/document.",
            "route_confidence": 0.99
        }

    if has_csv and any(word in question for word in csv_reference_words):
        return {
            "route": "csv_analyzer",
            "route_reason": "A CSV is uploaded and the user is referring to the current dataset.",
            "route_confidence": 0.99
        }

    if any(word in question for word in web_reference_words):
        return {
            "route": "web_search",
            "route_reason": "The user is asking for live/current web information.",
            "route_confidence": 0.99
        }

    if re.search(r"\d+\s*[\+\-\*/×÷]\s*\d+", question):
        return {
            "route": "calculator",
            "route_reason": "The user entered a clear arithmetic expression.",
            "route_confidence": 0.99
        }

    return None


# -----------------------------
# Router
# Comes from: Ollama/qwen3:8b + deterministic rules
# Role: decides which node/tool should handle the request
# -----------------------------
def router_node(state: AumState):
    override = deterministic_route_override(state)

    if override:
        return override

    router_prompt = """
You are a routing classifier for AUM State.

Choose exactly one route:
- spiritual
- technical
- business
- calculator
- csv_analyzer
- pdf_reader
- web_search

Route rules:
- calculator: calculate, compute, arithmetic, percentage, math expressions
- csv_analyzer: CSV, uploaded CSV, sales data, analyze data, dataframe, rows, columns, revenue, units
- pdf_reader: PDF, uploaded PDF, document, summarize document, explain document, policy, report, benefits, contract, chapter, preface, table of contents
- web_search: latest, current, live web, internet, news, price, search, browse, today, recently, current version, current docs
- spiritual: Gita, yoga, meditation, mantra, dharma, karma, spirituality, reflection
- technical: SQL, Spark, Python, code, errors, AI, GPU, NVIDIA, CUDA, WSL, Ollama, LangGraph
- business: money, career growth, business ideas, investments, productivity, decision-making

Return ONLY valid JSON:
{
  "route": "technical",
  "reason": "The user is asking about a software topic.",
  "confidence": 0.95
}
"""

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": state["question"]}
        ]
    )

    raw = response["message"]["content"].strip()

    try:
        data = json.loads(raw)
        route = data.get("route", "business").lower()
        reason = data.get("reason", "No reason provided.")
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        route = "business"
        reason = f"Router returned invalid JSON. Raw response: {raw}"
        confidence = 0.3

    if route not in TOOL_REGISTRY:
        route = "business"
        reason = "Router returned unknown route, defaulted to business."
        confidence = 0.3

    return {
        "route": route,
        "route_reason": reason,
        "route_confidence": confidence
    }


# -----------------------------
# Tool: Calculator
# Comes from: Python
# Role: exact arithmetic instead of model guessing
# -----------------------------
def calculator_node(state: AumState):
    question = state["question"]

    expression = question.lower()
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("plus", "+")
    expression = expression.replace("minus", "-")
    expression = expression.replace("times", "*")
    expression = expression.replace("multiplied by", "*")
    expression = expression.replace("divided by", "/")

    # Keep "x" only useful as multiplication after numbers/spaces.
    expression = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expression)

    expression = re.sub(r"[^0-9+\-*/().% ]", "", expression)
    expression = expression.replace("%", "/100")

    try:
        result = eval(expression, {"__builtins__": {}})
        answer = f"Calculator result: {result}"
        tool_result = f"Calculated expression: {expression}"
    except Exception:
        answer = "I could not calculate that safely. Try a clearer expression like: 25 * 4 + 10."
        tool_result = f"Calculator failed. Parsed expression: {expression}"

    return {
        "answer": answer,
        "tool_result": tool_result,
        "messages": update_memory(state, answer)
    }


# -----------------------------
# Tool: CSV Analyzer
# Upload: Streamlit file_uploader
# Reading: pandas
# Role: read real uploaded CSV and let Ollama explain summary
# -----------------------------
def csv_analyzer_node(state: AumState):
    csv_summary = state.get("csv_summary", "")

    if not csv_summary:
        answer = "CSV analyzer was selected, but no CSV file is uploaded yet. Upload a CSV file first, then ask me to analyze it."
        return {
            "answer": answer,
            "tool_result": "CSV analyzer selected, but no file found.",
            "messages": update_memory(state, answer)
        }

    prompt = """
You are a senior data analyst.

Analyze the CSV summary below.
Find useful business/data insights.
Be practical and concise.

Focus on:
- row count and columns
- numeric patterns
- possible sales/revenue/unit insights
- anomalies if visible
- what the user should check next

Do not invent columns or values that are not present.
"""

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"User question: {state['question']}\n\nCSV summary:\n{csv_summary}"
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "tool_result": "CSV analyzer used pandas summary + Ollama interpretation.",
        "messages": update_memory(state, answer)
    }


# -----------------------------
# Tool: PDF Reader with Intelligence + Simple RAG
# Upload: Streamlit file_uploader
# Extraction: pypdf
# RAG: Python chunking + keyword retrieval
# Role: answer from relevant PDF chunks, not only beginning of PDF
# -----------------------------
def pdf_reader_node(state: AumState):
    pdf_text = state.get("pdf_text", "")
    pdf_intelligence = state.get("pdf_intelligence", "")
    pdf_chunks = state.get("pdf_chunks", [])

    if not pdf_text:
        answer = "PDF reader was selected, but no PDF file is uploaded yet. Upload a PDF file first, then ask me to summarize or explain it."
        return {
            "answer": answer,
            "tool_result": "PDF reader selected, but no file found.",
            "messages": update_memory(state, answer)
        }

    relevant_chunks = retrieve_relevant_pdf_chunks(
        question=state["question"],
        chunks=pdf_chunks,
        top_k=5
    )

    if relevant_chunks:
        context_parts = []

        for chunk in relevant_chunks:
            context_parts.append(
                f"\n--- Relevant chunk from page {chunk['page']} | score {chunk['score']} ---\n{chunk['text']}"
            )

        rag_context = "\n".join(context_parts)
        rag_mode = "RAG mode: relevant PDF chunks retrieved."
    else:
        rag_context = pdf_text[:12000]
        rag_mode = "Fallback mode: no strong chunk match, used beginning of PDF."

    prompt = """
You are a careful PDF document analyst.

Use the PDF intelligence and retrieved PDF context below to answer the user's question.

Rules:
- Use only the PDF content provided.
- Do not invent facts.
- If the PDF does not contain the answer, say that clearly.
- Prefer structured answers.
- Mention page numbers when available.
- If the document looks like a table of contents/preface/sample chapter, say that clearly.
"""

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"User question: {state['question']}\n\n"
                    f"PDF intelligence:\n{pdf_intelligence[:8000]}\n\n"
                    f"{rag_mode}\n\n"
                    f"Retrieved PDF context:\n{rag_context}"
                )
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "tool_result": f"{rag_mode} Chunks used: {len(relevant_chunks) if relevant_chunks else 0}",
        "messages": update_memory(state, answer)
    }


# -----------------------------
# Web Search helper
# Comes from: ddgs Python package
# Role: fetch live web search results with title, URL, snippet
# -----------------------------
def run_web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=max_results
            )

            for item in search_results:
                title = item.get("title", "No title")
                url = item.get("href", "")
                snippet = item.get("body", "")

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })

        return results

    except Exception as e:
        return [{
            "title": "Web search failed",
            "url": "",
            "snippet": str(e)
        }]


def format_web_sources_for_llm(web_sources: List[Dict[str, str]]) -> str:
    if not web_sources:
        return "No web search results found."

    parts = []

    for idx, source in enumerate(web_sources, start=1):
        parts.append(
            f"Source {idx}:\n"
            f"Title: {source.get('title', '')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Snippet: {source.get('snippet', '')}\n"
        )

    return "\n".join(parts)


# -----------------------------
# Tool: Web Search
# Search: ddgs
# Role: retrieve current web results, save sources, then Ollama summarizes them
# -----------------------------
def web_search_node(state: AumState):
    query = state["question"]

    web_sources = run_web_search(query, max_results=5)

    if not web_sources:
        answer = "I searched the web, but no useful results were found."

        return {
            "answer": answer,
            "tool_result": "Web search returned no results.",
            "web_sources": [],
            "messages": update_memory(state, answer)
        }

    if web_sources[0]["title"] == "Web search failed":
        answer = (
            "I tried to search the web, but the search failed.\n\n"
            f"Details: {web_sources[0]['snippet']}"
        )

        return {
            "answer": answer,
            "tool_result": "Web search failed.",
            "web_sources": web_sources,
            "messages": update_memory(state, answer)
        }

    search_context = format_web_sources_for_llm(web_sources)

    prompt = """
You are a careful web research assistant.

Answer the user's question using only the search results below.

Rules:
- Do not claim you visited full web pages.
- Say the answer is based on search result snippets.
- Mention source titles when useful.
- Include URLs only when useful.
- If the results are weak, unclear, or conflicting, say that.
- Be concise and practical.
"""

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"User question: {state['question']}\n\n"
                    f"Web search sources:\n{search_context}"
                )
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "tool_result": f"Web search used {len(web_sources)} ddgs results.",
        "web_sources": web_sources,
        "messages": update_memory(state, answer)
    }


# -----------------------------
# Normal Ollama nodes
# Role: regular mode-specific assistant responses
# -----------------------------
def call_ollama(system_prompt: str, state: AumState):
    saved_facts = load_user_facts()

    memory_prompt = f"""
You have access to saved persistent user facts.

Saved facts:
{saved_facts}

Use these facts when relevant.
Do not say you have no memory if saved facts are provided.
"""

    messages = [{"role": "system", "content": system_prompt + "\n\n" + memory_prompt}]
    messages.extend(state["messages"])
    messages.append({"role": "user", "content": state["question"]})

    response = ollama.chat(
        model="qwen3:8b",
        messages=messages
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "messages": update_memory(state, answer)
    }


def spiritual_node(state: AumState):
    prompt = "You explain Indian spiritual concepts calmly, respectfully, and practically. Do not predict future or claim miracles."
    return call_ollama(prompt, state)


def technical_node(state: AumState):
    prompt = "You are a senior data engineering and AI systems mentor. Explain clearly with examples, especially SQL, Spark, Python, GPUs, WSL, Ollama, LangGraph, and architecture."
    return call_ollama(prompt, state)


def business_node(state: AumState):
    prompt = "You help small business owners think clearly, reduce risk, and take practical action."
    return call_ollama(prompt, state)


def video_plan_to_json(plan: VideoPlan) -> str:
    return json.dumps(asdict(plan), indent=2)


def video_creator_node(state: AumState):
    idea = state.get("video_request", "").strip()
    content_type = state.get("video_content_type", "").strip() or "nursery"

    if not idea:
        answer = "Video creator needs a video idea before it can build a storyboard."
        return {
            "answer": answer,
            "tool_result": "Video creator selected without a video idea.",
            "messages": update_memory(state, answer)
        }

    default_settings = build_video_settings(content_type=content_type)
    plan, warning = build_video_plan(idea=idea, settings=default_settings)
    answer = video_plan_to_json(plan)
    tool_result = warning or "Video creator generated a storyboard."

    return {
        "answer": answer,
        "tool_result": tool_result,
        "video_plan_json": answer,
        "messages": update_memory(state, answer)
    }


# -----------------------------
# Tool Registry
# Comes from: Python dictionary
# Role: central route/tool registration
# -----------------------------
TOOL_REGISTRY: Dict[str, Callable] = {
    "spiritual": spiritual_node,
    "technical": technical_node,
    "business": business_node,
    "calculator": calculator_node,
    "csv_analyzer": csv_analyzer_node,
    "pdf_reader": pdf_reader_node,
    "web_search": web_search_node,
    "video_creator": video_creator_node,
}


def route_decision(state: AumState):
    return state["route"]


# -----------------------------
# CSV helper
# Comes from: pandas
# Role: summarize real uploaded CSV
# -----------------------------
def build_csv_summary(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    df = pd.read_csv(uploaded_file)

    row_count = len(df)
    column_count = len(df.columns)
    columns = list(df.columns)

    numeric_df = df.select_dtypes(include="number")

    summary_parts = []
    summary_parts.append(f"Rows: {row_count}")
    summary_parts.append(f"Columns count: {column_count}")
    summary_parts.append(f"Columns: {columns}")

    if not numeric_df.empty:
        summary_parts.append("\nNumeric summary:")
        summary_parts.append(numeric_df.describe().to_string())

    summary_parts.append("\nFirst 5 rows:")
    summary_parts.append(df.head(5).to_string(index=False))

    return "\n".join(summary_parts)


# -----------------------------
# PDF helpers
# Comes from: pypdf + Python regex/rules
# Role: extract pages, build intelligence, chunk text for RAG
# -----------------------------
def extract_pdf_pages(uploaded_file) -> List[Dict[str, str]]:
    if uploaded_file is None:
        return []

    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append({
            "page": str(page_number),
            "text": page_text.strip()
        })

    return pages


def build_pdf_text_from_pages(pages: List[Dict[str, str]]) -> str:
    text_parts = []

    for page in pages:
        if page["text"]:
            text_parts.append(f"\n--- Page {page['page']} ---\n{page['text']}")

    return "\n".join(text_parts)


def detect_possible_title(pages: List[Dict[str, str]]) -> str:
    if not pages:
        return "Unknown"

    first_text = pages[0]["text"]
    lines = [line.strip() for line in first_text.splitlines() if line.strip()]

    for line in lines[:10]:
        if 5 <= len(line) <= 120:
            return line

    return "Unknown"


def detect_document_type(full_text: str) -> str:
    text = full_text.lower()

    if "table of contents" in text and "preface" in text:
        return "Book front matter / table of contents / preface"
    if "table of contents" in text:
        return "Document with table of contents"
    if "preface" in text:
        return "Book/document preface"
    if "chapter" in text:
        return "Book or chapter-based document"
    if "benefits" in text or "coverage" in text or "deductible" in text:
        return "Benefits / policy document"
    if "agreement" in text or "contract" in text:
        return "Agreement / contract document"
    if "invoice" in text or "amount due" in text:
        return "Invoice / billing document"
    if "resume" in text or ("experience" in text and "skills" in text):
        return "Resume / profile document"

    return "General PDF document"


def extract_toc_candidates(pages: List[Dict[str, str]]) -> List[str]:
    candidates = []

    chapter_regex = r"\bchapter\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\b"

    for page in pages[:15]:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]

        for i, line in enumerate(lines):
            lower = line.lower()

            if len(line) > 160:
                continue

            if "table of contents" in lower:
                candidates.append(f"Page {page['page']}: {line}")
                continue

            if re.search(chapter_regex, lower):
                combined = line

                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if 3 <= len(next_line) <= 120:
                        combined = f"{line} - {next_line}"

                candidates.append(f"Page {page['page']}: {combined}")
                continue

            if re.search(r"^\d+(\.\d+)*\.?\s+[A-Za-z]", line):
                candidates.append(f"Page {page['page']}: {line}")
                continue

            if re.search(r"\.{2,}\s*\d+$", line):
                candidates.append(f"Page {page['page']}: {line}")
                continue

            if any(word in lower for word in ["preface", "introduction", "appendix", "index"]):
                if len(line) <= 120:
                    candidates.append(f"Page {page['page']}: {line}")
                    continue

    seen = set()
    unique_candidates = []

    for item in candidates:
        if item not in seen:
            unique_candidates.append(item)
            seen.add(item)

    return unique_candidates[:80]


def extract_section_candidates(pages: List[Dict[str, str]]) -> List[str]:
    sections = []

    general_heading_words = [
        "preface",
        "introduction",
        "chapter",
        "appendix",
        "summary",
        "overview",
        "getting started",
        "installation",
        "configuration",
        "setup",
        "conclusion",
        "contents",
        "table of contents",
        "index",
        "references",
        "about the author",
        "acknowledgements",
        "requirements",
        "examples",
        "notes"
    ]

    chapter_regex = r"\bchapter\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\b"

    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]

        for i, line in enumerate(lines):
            lower = line.lower()

            if len(line) > 160:
                continue

            if re.search(chapter_regex, lower):
                combined = line

                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if 3 <= len(next_line) <= 120:
                        combined = f"{line} - {next_line}"

                sections.append(f"Page {page['page']}: {combined}")
                continue

            if re.search(r"\bpart\s+(\d+|i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", lower):
                sections.append(f"Page {page['page']}: {line}")
                continue

            if re.search(r"\bsection\s+\d+(\.\d+)*\b", lower):
                sections.append(f"Page {page['page']}: {line}")
                continue

            if re.search(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z]", line):
                sections.append(f"Page {page['page']}: {line}")
                continue

            if re.search(r"\bappendix\s+([a-z]|\d+)\b", lower):
                sections.append(f"Page {page['page']}: {line}")
                continue

            if any(word in lower for word in general_heading_words):
                sections.append(f"Page {page['page']}: {line}")
                continue

            if line.isupper() and 4 <= len(line) <= 100:
                noisy_words = ["isbn", "copyright", "published", "mumbai", "birmingham"]
                if not any(noisy in lower for noisy in noisy_words):
                    sections.append(f"Page {page['page']}: {line}")
                continue

    seen = set()
    unique_sections = []

    for section in sections:
        if section not in seen:
            unique_sections.append(section)
            seen.add(section)

    return unique_sections[:100]


def build_pdf_intelligence(pages: List[Dict[str, str]]) -> str:
    if not pages:
        return ""

    full_text = build_pdf_text_from_pages(pages)

    page_count = len(pages)
    title = detect_possible_title(pages)
    document_type = detect_document_type(full_text)
    total_chars = len(full_text)

    non_empty_pages = [p for p in pages if p["text"].strip()]
    empty_page_count = page_count - len(non_empty_pages)

    toc_candidates = extract_toc_candidates(pages)
    section_candidates = extract_section_candidates(pages)

    intelligence_parts = []

    intelligence_parts.append("PDF INTELLIGENCE REPORT")
    intelligence_parts.append(f"Possible title: {title}")
    intelligence_parts.append(f"Detected document type: {document_type}")
    intelligence_parts.append(f"Total pages: {page_count}")
    intelligence_parts.append(f"Pages with extracted text: {len(non_empty_pages)}")
    intelligence_parts.append(f"Pages with no extracted text: {empty_page_count}")
    intelligence_parts.append(f"Total extracted characters: {total_chars}")

    if toc_candidates:
        intelligence_parts.append("\nPossible table of contents / chapter entries:")
        for item in toc_candidates:
            intelligence_parts.append(f"- {item}")
    else:
        intelligence_parts.append("\nPossible table of contents / chapter entries: Not detected")

    if section_candidates:
        intelligence_parts.append("\nPossible section/headings:")
        for item in section_candidates:
            intelligence_parts.append(f"- {item}")
    else:
        intelligence_parts.append("\nPossible section/headings: Not detected")

    if empty_page_count > 0:
        intelligence_parts.append(
            "\nWarning: Some pages had no extractable text. "
            "The PDF may contain scanned images or image-based pages."
        )

    return "\n".join(intelligence_parts)


def chunk_pdf_pages(pages: List[Dict[str, str]], chunk_size: int = 1200, overlap: int = 200) -> List[Dict[str, str]]:
    chunks = []

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "page": page_number,
                "text": chunk_text
            })

            start = end - overlap

            if start < 0:
                start = 0

            if start >= len(text):
                break

    return chunks


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9]+", text)


def retrieve_relevant_pdf_chunks(question: str, chunks: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, str]]:
    if not chunks:
        return []

    question_tokens = set(tokenize(question))

    scored_chunks = []

    for chunk in chunks:
        chunk_tokens = tokenize(chunk["text"])

        if not chunk_tokens:
            continue

        chunk_token_set = set(chunk_tokens)
        overlap_score = len(question_tokens.intersection(chunk_token_set))

        phrase_bonus = 0
        if question.lower() in chunk["text"].lower():
            phrase_bonus = 5

        score = overlap_score + phrase_bonus

        scored_chunks.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "score": score
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return [chunk for chunk in scored_chunks[:top_k] if chunk["score"] > 0]


# -----------------------------
# LangGraph
# Comes from: LangGraph
# Role: workflow engine for routing request to correct node/tool
# -----------------------------
builder = StateGraph(AumState)

builder.add_node("router", router_node)

for tool_name, tool_function in TOOL_REGISTRY.items():
    builder.add_node(tool_name, tool_function)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    route_decision,
    {tool_name: tool_name for tool_name in TOOL_REGISTRY.keys()}
)

for tool_name in TOOL_REGISTRY.keys():
    builder.add_edge(tool_name, END)

graph = builder.compile()


# -----------------------------
# Streamlit UI
# Comes from: Streamlit
# Role: browser interface for AUM State
# -----------------------------
st.set_page_config(page_title="AUM State", page_icon="ॐ")
st.title("AUM State")
st.caption("AI for clarity, work, and wisdom")

init_memory_db()

if "messages" not in st.session_state:
    st.session_state.messages = load_messages_from_db(limit=30)

if "csv_summary" not in st.session_state:
    st.session_state.csv_summary = ""

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "pdf_intelligence" not in st.session_state:
    st.session_state.pdf_intelligence = ""

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "web_sources" not in st.session_state:
    st.session_state.web_sources = []

if "video_plan" not in st.session_state:
    st.session_state.video_plan = None

if "video_plan_warning" not in st.session_state:
    st.session_state.video_plan_warning = ""

if "video_output_dir" not in st.session_state:
    st.session_state.video_output_dir = ""

if "video_image_paths" not in st.session_state:
    st.session_state.video_image_paths = []

if "video_audio_path" not in st.session_state:
    st.session_state.video_audio_path = ""

if "video_file_path" not in st.session_state:
    st.session_state.video_file_path = ""

if "video_status_messages" not in st.session_state:
    st.session_state.video_status_messages = []

if "video_quality_label" not in st.session_state:
    st.session_state.video_quality_label = DEFAULT_IMAGE_QUALITY

if "video_settings_snapshot" not in st.session_state:
    st.session_state.video_settings_snapshot = ""


st.sidebar.header("Tools")

# Feature: CSV upload
# Comes from: Streamlit file_uploader
# Role: lets user upload CSV from browser
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if csv_file is not None:
    try:
        st.session_state.csv_summary = build_csv_summary(csv_file)
        st.sidebar.success("CSV loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"CSV load failed: {e}")
        st.session_state.csv_summary = ""


# Feature: PDF upload
# Comes from: Streamlit file_uploader
# Role: lets user upload PDF from browser
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if pdf_file is not None:
    try:
        pdf_pages = extract_pdf_pages(pdf_file)
        st.session_state.pdf_text = build_pdf_text_from_pages(pdf_pages)
        st.session_state.pdf_intelligence = build_pdf_intelligence(pdf_pages)
        st.session_state.pdf_chunks = chunk_pdf_pages(pdf_pages)

        if st.session_state.pdf_text.strip():
            st.sidebar.success(f"PDF loaded with intelligence. Chunks: {len(st.session_state.pdf_chunks)}")
        else:
            st.sidebar.warning("PDF loaded, but no text was extracted. It may be scanned/image-based.")
    except Exception as e:
        st.sidebar.error(f"PDF load failed: {e}")
        st.session_state.pdf_text = ""
        st.session_state.pdf_intelligence = ""
        st.session_state.pdf_chunks = []


with st.sidebar.expander("PDF Intelligence Preview"):
    if st.session_state.pdf_intelligence:
        st.text(st.session_state.pdf_intelligence[:3000])
    else:
        st.caption("Upload a PDF to see document intelligence.")


with st.sidebar.expander("Last Web Search Sources"):
    if st.session_state.web_sources:
        for idx, source in enumerate(st.session_state.web_sources, start=1):
            title = source.get("title", "No title")
            url = source.get("url", "")
            snippet = source.get("snippet", "")

            if url:
                st.markdown(f"**{idx}. [{title}]({url})**")
            else:
                st.markdown(f"**{idx}. {title}**")

            if snippet:
                st.caption(snippet)
    else:
        st.caption("No web search sources yet.")


question = st.text_area("Ask AUM State:")

col1, col2 = st.columns(2)

with col1:
    ask_clicked = st.button("Ask")

with col2:
    clear_clicked = st.button("Clear Memory")

if clear_clicked:
    st.session_state.messages = []
    clear_memory_db()
    clear_user_facts()
    st.success("Memory cleared from session, chat history, and saved facts.")

if ask_clicked and question:
    with st.spinner("Thinking..."):
        result = graph.invoke({
            "question": question,
            "route": "",
            "route_reason": "",
            "route_confidence": 0.0,
            "answer": "",
            "messages": st.session_state.messages,
            "tool_result": "",
            "csv_summary": st.session_state.csv_summary,
            "pdf_text": st.session_state.pdf_text,
            "pdf_intelligence": st.session_state.pdf_intelligence,
            "pdf_chunks": st.session_state.pdf_chunks,
            "web_sources": st.session_state.web_sources,
            "video_request": "",
            "video_content_type": "",
            "video_plan_json": "",
            "video_output_dir": "",
            "video_image_paths": [],
            "video_audio_path": "",
            "video_file_path": "",
            "video_status_messages": []
        })

        st.session_state.messages = result["messages"]

        if "web_sources" in result and result["web_sources"]:
            st.session_state.web_sources = result["web_sources"]

        st.caption(f"Routed to: {result['route']}")
        st.caption(f"Reason: {result['route_reason']}")
        st.caption(f"Confidence: {round(result['route_confidence'] * 100, 1)}%")

        if result.get("tool_result"):
            st.caption(f"Tool: {result['tool_result']}")

        st.write(result["answer"])


st.divider()

st.subheader("Video Studio")
st.caption("Standalone storyboard and rendering workflow. Chat routing remains unchanged.")

video_idea = st.text_area(
    "Video idea",
    placeholder="Create a 15-second vertical nursery video about a baby elephant learning colors.",
    key="video_idea_input"
)
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    video_duration_label = st.selectbox(
        "Video duration",
        options=["10 seconds", "15 seconds", "30 seconds", "45 seconds", "60 seconds"],
        index=["10 seconds", "15 seconds", "30 seconds", "45 seconds", "60 seconds"].index(f"{DEFAULT_TOTAL_DURATION_SECONDS} seconds"),
        key="video_duration_label",
    )
with row1_col2:
    scene_duration_label = st.selectbox(
        "Seconds per scene",
        options=["3 seconds", "4 seconds", "5 seconds", "6 seconds"],
        index=["3 seconds", "4 seconds", "5 seconds", "6 seconds"].index(f"{DEFAULT_SCENE_DURATION_SECONDS} seconds"),
        key="scene_duration_label",
    )
with row1_col3:
    frame_rate_label = st.selectbox(
        "Frame rate",
        options=["24 FPS", "30 FPS"],
        index=["24 FPS", "30 FPS"].index(f"{DEFAULT_FRAME_RATE} FPS"),
        key="frame_rate_label",
    )

row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    video_content_type = st.selectbox(
        "Content type",
        options=["nursery", "education", "story", "explainer"],
        index=0,
        key="video_content_type_input"
    )
with row2_col2:
    video_quality = st.selectbox(
        "Image quality",
        options=["Draft", "Standard", "Final"],
        index=["Draft", "Standard", "Final"].index(st.session_state.video_quality_label)
        if st.session_state.video_quality_label in ["Draft", "Standard", "Final"]
        else 0,
        key="video_quality_label"
    )
with row2_col3:
    visual_style = st.selectbox(
        "Visual style",
        options=["3D Nursery Animation", "Soft Watercolor", "Storybook Illustration"],
        index=["3D Nursery Animation", "Soft Watercolor", "Storybook Illustration"].index(DEFAULT_VISUAL_STYLE),
        key="visual_style_label",
    )

st.caption("Draft: cheapest, best for testing")
st.caption("Standard: balanced quality and cost")
st.caption("Final: highest quality, use only for approved videos")

row3_col1, row3_col2, row3_col3 = st.columns(3)
with row3_col1:
    format_label = st.selectbox(
        "Format",
        options=["Vertical 9:16", "Landscape 16:9", "Square 1:1"],
        index=["Vertical 9:16", "Landscape 16:9", "Square 1:1"].index(DEFAULT_ASPECT_RATIO_LABEL),
        key="format_label",
    )
with row3_col2:
    motion_label = st.selectbox(
        "Motion",
        options=["Still", "Gentle"],
        index=["Still", "Gentle"].index(DEFAULT_MOTION_LEVEL),
        key="motion_label",
    )
with row3_col3:
    narration_enabled = st.selectbox(
        "Narration",
        options=["On", "Off"],
        index=0 if DEFAULT_NARRATION_ENABLED else 1,
        key="narration_enabled_label",
    )

language = DEFAULT_LANGUAGE
voice = DEFAULT_VOICE
speaking_style = DEFAULT_SPEAKING_STYLE
speaking_speed = DEFAULT_SPEAKING_SPEED

if narration_enabled == "On":
    row4_col1, row4_col2, row4_col3, row4_col4 = st.columns(4)
    with row4_col1:
        language = st.selectbox(
            "Language",
            options=["English", "Hindi", "Telugu"],
            index=["English", "Hindi", "Telugu"].index(DEFAULT_LANGUAGE),
            key="language_label",
        )
    with row4_col2:
        voice = st.selectbox(
            "Voice",
            options=["Warm Female", "Warm Male", "Neutral"],
            index=["Warm Female", "Warm Male", "Neutral"].index(DEFAULT_VOICE),
            key="voice_label",
        )
    with row4_col3:
        speaking_style = st.selectbox(
            "Speaking style",
            options=["Warm", "Playful", "Calm"],
            index=["Warm", "Playful", "Calm"].index(DEFAULT_SPEAKING_STYLE),
            key="speaking_style_label",
        )
    with row4_col4:
        speaking_speed = st.selectbox(
            "Speaking speed",
            options=["Slow", "Normal", "Fast"],
            index=["Slow", "Normal", "Fast"].index(DEFAULT_SPEAKING_SPEED),
            key="speaking_speed_label",
        )

current_video_settings = build_video_settings(
    total_duration_seconds=int(video_duration_label.split()[0]),
    preferred_scene_duration_seconds=int(scene_duration_label.split()[0]),
    frame_rate=int(frame_rate_label.split()[0]),
    aspect_ratio_label=format_label,
    content_type=video_content_type,
    visual_style=visual_style,
    image_quality=video_quality,
    motion_level=motion_label,
    narration_enabled=narration_enabled == "On",
    language=language,
    voice=voice,
    speaking_style=speaking_style,
    speaking_speed=speaking_speed,
)

storyboard_stale = (
    bool(st.session_state.video_plan)
    and bool(st.session_state.video_settings_snapshot)
    and st.session_state.video_settings_snapshot != settings_snapshot(current_video_settings)
)

st.markdown("**Generation Summary**")
st.caption(f"Duration: {current_video_settings.total_duration_seconds} seconds")
st.caption(f"Scenes: {current_video_settings.scene_count}")
st.caption(
    "Scene timing: "
    + ", ".join(f"{duration}s" for duration in current_video_settings.scene_durations)
)
st.caption(f"Frame rate: {current_video_settings.frame_rate} FPS")
st.caption(
    f"Format: {format_label.split()[0]} {current_video_settings.output_width}x{current_video_settings.output_height}"
)
st.caption(f"Content type: {current_video_settings.content_type}")
st.caption(f"Image quality: {current_video_settings.image_quality}")
if current_video_settings.narration_enabled:
    st.caption(
        f"Narration: {current_video_settings.language}, {current_video_settings.voice}, {current_video_settings.speaking_style}"
    )
else:
    st.caption("Narration: Off")

video_col1, video_col2 = st.columns(2)

with video_col1:
    storyboard_clicked = st.button("Create storyboard")

with video_col2:
    generate_video_clicked = st.button("Generate assets and video")

if storyboard_clicked:
    if not video_idea.strip():
        st.warning("Enter a video idea before creating a storyboard.")
    else:
        with st.spinner("Creating storyboard..."):
            video_plan, video_warning = build_video_plan(
                idea=video_idea,
                settings=current_video_settings
            )
            st.session_state.video_plan = video_plan
            st.session_state.video_plan_warning = video_warning or ""
            st.session_state.video_settings_snapshot = settings_snapshot(current_video_settings)
            st.session_state.video_output_dir = ""
            st.session_state.video_image_paths = []
            st.session_state.video_audio_path = ""
            st.session_state.video_file_path = ""
            st.session_state.video_status_messages = []

if st.session_state.video_plan_warning:
    st.info(st.session_state.video_plan_warning)

if storyboard_stale:
    st.warning("Settings changed. Create a new storyboard before generating assets.")

if st.session_state.video_plan:
    current_video_plan = st.session_state.video_plan
    st.markdown(f"**Title:** {current_video_plan.title}")
    st.markdown(f"**Narration:** {current_video_plan.narration}")
    st.markdown(f"**Style Lock:** {current_video_plan.style_lock}")
    st.markdown("**Storyboard**")

    for scene in current_video_plan.scenes:
        with st.container(border=True):
            st.markdown(f"**Scene {scene.scene_number}** ({scene.duration_seconds}s)")
            st.write(scene.narration)
            st.caption(f"Visual prompt: {scene.visual_prompt}")
            st.caption(f"Motion prompt: {scene.motion_prompt}")

if generate_video_clicked:
    if not st.session_state.video_plan:
        st.warning("Create a storyboard before generating assets and video.")
    elif storyboard_stale:
        st.warning("Settings changed. Create a new storyboard before generating assets.")
    else:
        progress_box = st.empty()

        try:
            progress_box.info("Checking FFmpeg...")
            ensure_ffmpeg_available()

            output_dir = build_generation_output_dir()
            st.session_state.video_output_dir = str(output_dir)

            progress_box.info("Generating scene images...")
            image_paths, image_messages = generate_scene_images(
                plan=st.session_state.video_plan,
                output_dir=output_dir,
                settings=current_video_settings
            )

            progress_box.info("Generating narration audio...")
            audio_path, audio_message = generate_narration_audio(
                plan=st.session_state.video_plan,
                output_dir=output_dir,
                settings=current_video_settings
            )

            progress_box.info("Rendering video...")
            video_path = render_video(
                image_paths=image_paths,
                scene_durations=[scene.duration_seconds for scene in st.session_state.video_plan.scenes],
                output_dir=output_dir,
                audio_path=audio_path,
                settings=current_video_settings
            )

            st.session_state.video_image_paths = [str(path) for path in image_paths]
            st.session_state.video_audio_path = str(audio_path) if audio_path else ""
            st.session_state.video_file_path = str(video_path)
            st.session_state.video_status_messages = image_messages + [audio_message, f"Rendered video: {video_path.name}"]
            progress_box.success("Video generation complete.")
        except Exception as e:
            progress_box.error(f"Video generation failed: {e}")

if st.session_state.video_status_messages:
    st.markdown("**Video generation status**")
    for message in st.session_state.video_status_messages:
        st.caption(message)

if st.session_state.video_image_paths:
    st.markdown("**Scene previews**")
    preview_columns = st.columns(len(st.session_state.video_image_paths))

    for index, image_path in enumerate(st.session_state.video_image_paths):
        with preview_columns[index]:
            st.image(image_path, caption=f"Scene {index + 1}", use_container_width=True)

if st.session_state.video_file_path:
    st.markdown("**Final video**")
    st.video(st.session_state.video_file_path)

    with open(st.session_state.video_file_path, "rb") as video_file:
        st.download_button(
            label="Download MP4",
            data=video_file,
            file_name="aumstate_video.mp4",
            mime="video/mp4"
        )

st.divider()

st.subheader("Memory / Chat History")
st.caption(f"Persistent memory file: {DB_PATH}")

st.subheader("Saved User Facts")
st.text(load_user_facts())

if st.session_state.messages:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AUM State:** {msg['content']}")
else:
    st.caption("No memory yet.")
