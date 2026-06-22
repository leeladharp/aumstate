import streamlit as st
import ollama
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END


class AumState(TypedDict):
    question: str
    route: str
    answer: str
    messages: List[Dict[str, str]]


def router_node(state: AumState):
    q = state["question"].lower()

    if any(word in q for word in [
        "gita", "karma", "yoga", "spiritual", "mantra", "meditation", "aum", "om"
    ]):
        return {"route": "spiritual"}

    elif any(word in q for word in [
        "sql", "spark", "python", "error", "code", "data", "gpu", "nvidia",
        "ollama", "wsl", "linux", "cuda", "langgraph", "api"
    ]):
        return {"route": "technical"}

    else:
        return {"route": "business"}


def call_ollama(system_prompt: str, state: AumState):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(state["messages"])
    messages.append({"role": "user", "content": state["question"]})

    response = ollama.chat(
        model="qwen3:8b",
        messages=messages
    )

    answer = response["message"]["content"]

    updated_messages = state["messages"] + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer}
    ]

    return {
        "answer": answer,
        "messages": updated_messages
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


def route_decision(state: AumState):
    return state["route"]


builder = StateGraph(AumState)

builder.add_node("router", router_node)
builder.add_node("spiritual", spiritual_node)
builder.add_node("technical", technical_node)
builder.add_node("business", business_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "spiritual": "spiritual",
        "technical": "technical",
        "business": "business"
    }
)

builder.add_edge("spiritual", END)
builder.add_edge("technical", END)
builder.add_edge("business", END)

graph = builder.compile()


st.set_page_config(page_title="AUM State", page_icon="ॐ")
st.title("AUM State")
st.caption("AI for clarity, work, and wisdom")

if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.text_area("Ask AUM State:")

col1, col2 = st.columns(2)

with col1:
    ask_clicked = st.button("Ask")

with col2:
    clear_clicked = st.button("Clear Memory")

if clear_clicked:
    st.session_state.messages = []
    st.success("Memory cleared.")

if ask_clicked and question:
    with st.spinner("Thinking..."):
        result = graph.invoke({
            "question": question,
            "route": "",
            "answer": "",
            "messages": st.session_state.messages
        })

        st.session_state.messages = result["messages"]

        st.caption(f"Routed to: {result['route']}")
        st.write(result["answer"])

st.divider()

st.subheader("Memory / Chat History")

if st.session_state.messages:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**AUM State:** {content}")
else:
    st.caption("No memory yet.")
