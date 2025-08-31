from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from Agentic_RAG.src.utils import load_embedding_model
from Agentic_RAG.src.retriver import get_milvus_retriever, get_retriever_tool
from Agentic_RAG.src.graph.nodes import (get_generate_query_or_respond_node, get_rewrite_question_node,
                                         get_generate_answer_node, get_grade_documents_node)


# Defines agent state and manages messages
class AgentState(TypedDict):
    # Manages the sequence of messages using the add_messages reducer function
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_graph(config: dict):
    # response_model = init_chat_model(model=config["llm"]["model"],
    #                                  model_provider=config["llm"]["model_provider"],
    #                                  api_key=config["llm"]["api_key"],
    #                                  base_url=config["llm"]["base_url"],
    #                                  temperature=config["llm"]["temperature"]
    #                                  )

    embedding_model = load_embedding_model(model_name=config["embedding_model"]["model_path"])
    retriever = get_milvus_retriever(database_name=config["database"]["db_name"],
                                     collection_name=config["database"]["collection_name"],
                                     embedding_model=embedding_model)

    retriever_tool = get_retriever_tool(retriever=retriever)

    workflow = StateGraph(AgentState)

    generate_query_or_respond_node = get_generate_query_or_respond_node(llm_config=config["llm"],
                                                                        retriever_tool=retriever_tool)
    rewrite_question_node = get_rewrite_question_node(llm_config=config["llm"])

    generate_answer_node = get_generate_answer_node(llm_config=config["llm"])
    grade_documents_node = get_grade_documents_node(llm_config=config["llm"])

    workflow.add_node("generate_query_or_respond_node", generate_query_or_respond_node)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question_node", rewrite_question_node)
    workflow.add_node("generate_answer_node", generate_answer_node)

    workflow.add_edge(START, "generate_query_or_respond_node")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond_node",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents_node,
    )

    workflow.add_edge("rewrite_question_node", "generate_query_or_respond_node")
    workflow.add_edge("generate_answer_node", END)

    graph = workflow.compile()

    plot = graph.get_graph(xray=True).draw_mermaid_png()
    with open("../images/rag_agent.png", "wb") as fp:
        fp.write(plot)

    return graph
