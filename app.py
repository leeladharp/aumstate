import streamlit as st
import ollama
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AumState(TypedDict):
    mode: str
    question: str
    answer: str


prompts = {
    "Spiritual Wisdom": "You explain Indian spiritual concepts calmly, respectfully, and practically. Do not claim miracles or predict the future.",
    "Technical Mentor": "You are a senior data engineering mentor. Explain with examples, especially SQL, Spark, Python, and architecture.",
    "Business Clarity": "You help small business owners think clearly, find opportunities, reduce risk, and take practical action."
}


def llm_node(state: AumState):
    mode = state["mode"]
    question = state["question"]

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": prompts[mode]},
            {"role": "user", "content": question}
        ]
    )

    return {"answer": response["message"]["content"]}


graph_builder = StateGraph(AumState)

graph_builder.add_node("llm_node", llm_node)

graph_builder.add_edge(START, "llm_node")
graph_builder.add_edge("llm_node", END)

graph = graph_builder.compile()


st.set_page_config(page_title="AUM State", page_icon="ॐ")

st.title("AUM State")
st.caption("AI for clarity, work, and wisdom")

mode = st.selectbox(
    "Choose mode:",
    ["Spiritual Wisdom", "Technical Mentor", "Business Clarity"]
)

question = st.text_area("Ask AUM State:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        result = graph.invoke({
            "mode": mode,
            "question": question,
            "answer": ""
        })

        st.write(result["answer"])
