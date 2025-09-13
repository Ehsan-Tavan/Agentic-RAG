from langgraph.graph import StateGraph, START, END

from Corrective_RAG.src.graph.state import GraphState
from Corrective_RAG.src.graph.nodes import get_retriever_node, get_document_grader_node, get_generation_node, \
    get_rewrite_query_node, get_web_search_node


def decide_to_generate(state: GraphState):
    # Determine the next step based on the evaluated documents
    print("==== [ASSESS GRADED DOCUMENTS] ====")
    # Whether web search is required
    web_search = state["web_search"]

    if web_search == "Yes":
        # When additional information is needed through web search
        print(
            "==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, QUERY REWRITE] ===="
        )
        # Route to the query-rewrite node
        return "query_rewrite"
    else:
        # Relevant documents exist, so proceed to the answer generation step (generate)
        print("==== [DECISION: GENERATE] ====")
        return "generate"


def create_graph(config: dict):
    workflow = StateGraph(GraphState)
    retriever_node = get_retriever_node(config=config)
    document_grader_node = get_document_grader_node(config=config)
    generation_node = get_generation_node(config=config)
    rewrite_query_node = get_rewrite_query_node(config=config)
    web_search_node= get_web_search_node()

    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("documents_grader", document_grader_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("query_rewrite", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "documents_grader")

    workflow.add_conditional_edges(
        "documents_grader",
        decide_to_generate,
        {
            "query_rewrite": "query_rewrite",
            "generate": "generation",
        }
    )

    workflow.add_edge("query_rewrite", "web_search")
    workflow.add_edge("web_search", "generation")
    workflow.add_edge("generation", END)

    app = workflow.compile()

    # plot = app.get_graph(xray=True).draw_mermaid_png()
    # with open("../images/search_agent.png", "wb") as fp:
    #     fp.write(plot)

    return app


