from langchain.prompts import PromptTemplate
from langgraph.graph import MessagesState
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def get_generate_answer_node(
        llm_config: dict,
):
    """Generate an answer."""
    model = ChatOpenAI(
        temperature=llm_config["temperature"],
        model=llm_config["model"],
        streaming=True)

    prompt_template = PromptTemplate(
        template=GENERATE_PROMPT,
        input_variables=["question", "context"]
    )

    def _generate_answer_node(state: MessagesState):
        print("==== [Generate Answer or Respond] ====")

        question = state["messages"][0].content
        context = state["messages"][-1].content

        rag_chain = prompt_template | model | StrOutputParser()

        response = rag_chain.invoke({"context": context, "question": question})

        return {"messages": [response]}

    return _generate_answer_node
