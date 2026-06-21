import streamlit as st
import ollama

st.set_page_config(page_title="AUM State", page_icon="ॐ")

st.title("AUM State")
st.caption("AI for clarity, work, and wisdom")

mode = st.selectbox(
    "Choose mode:",
    ["Spiritual Wisdom", "Technical Mentor", "Business Clarity"]
)

question = st.text_area("Ask AUM State:")

prompts = {
    "Spiritual Wisdom": "You explain Indian spiritual concepts calmly, respectfully, and practically. Do not claim miracles or predict the future.",
    "Technical Mentor": "You are a senior data engineering mentor. Explain with examples, especially SQL, Spark, Python, and architecture.",
    "Business Clarity": "You help small business owners think clearly, find opportunities, reduce risk, and take practical action."
}

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = ollama.chat(
            model="qwen3:8b",
            messages=[
                {"role": "system", "content": prompts[mode]},
                {"role": "user", "content": question}
            ]
        )
        st.write(response["message"]["content"])
