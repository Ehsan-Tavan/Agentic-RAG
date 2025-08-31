from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI


REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def get_rewrite_question_node(
        llm_config: dict
):
    """Rewrite the original user question."""
    model = ChatOpenAI(temperature=llm_config["temperature"], model=llm_config["model"], streaming=True)

    def _rewrite_question_node(state: MessagesState):
        print("==== [QUERY REWRITE] ====")
        messages = state["messages"]
        question = messages[0].content

        prompt = REWRITE_PROMPT.format(question=question)
        response = model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}
    return _rewrite_question_node