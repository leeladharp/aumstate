from langgraph.graph import StateGraph, MessagesState, START, END
import ollama


def chat(state: MessagesState):

    response = ollama.chat(
        model="qwen3:8b",
        messages=state["messages"]
    )

    return {
        "messages": [
            response["message"]
        ]
    }


graph = StateGraph(MessagesState)

graph.add_node("chat", chat)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile()

result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain Karma Yoga"
            }
        ]
    }
)

print(result["messages"][-1].content)
