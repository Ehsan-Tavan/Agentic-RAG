from langgraph.graph import MessagesState
from langchain_core.language_models.chat_models import BaseChatModel


def get_generate_query_or_respond_node(response_model: BaseChatModel, retriever_tool):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """

    def _generate_query_or_respond_node(state: MessagesState):
        response = (
            response_model
            .bind_tools([retriever_tool]).invoke(state["messages"])
        )
        return {"messages": [response]}

    return _generate_query_or_respond_node
