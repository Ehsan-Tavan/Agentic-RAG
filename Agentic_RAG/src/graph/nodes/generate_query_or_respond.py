from langchain_core.tools.simple import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState


def get_generate_query_or_respond_node(
        llm_config: dict,
        retriever_tool: Tool
):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    model = ChatOpenAI(temperature=llm_config["temperature"], model=llm_config["model"], streaming=True)

    # Bind the retriever tool
    model = model.bind_tools([retriever_tool])

    def _generate_query_or_respond_node(
            state: MessagesState
    ):
        print("==== [Generate Query or Respond] ====")

        messages = state["messages"]

        # Generate agent response
        response = model.invoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("=== Model is calling a tool ===")
            print(response.tool_calls)
        else:
            print("=== Model is answering directly ===")
            print(response.content)
        return {"messages": [response]}

    return _generate_query_or_respond_node
