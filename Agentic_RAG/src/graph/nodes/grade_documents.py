from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Response 'yes' if the document is relevant to the question or 'no' if it is not."
    )


def get_grade_documents_node(
        llm_config: dict
):
    grader_model = ChatOpenAI(temperature=llm_config["temperature"], model=llm_config["model"], streaming=True)
    grader_model = grader_model.with_structured_output(GradeDocuments)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | grader_model

    def _grade_documents_node(state: MessagesState) -> Literal["generate_answer_node", "rewrite_question_node"]:
        """Determine whether the retrieved documents are relevant to the question."""
        print("==== [Grade Documents] ====")

        # Extract messages from the current state
        messages = state["messages"]

        # Get the most recent message
        last_message = messages[-1]

        # Extract the original question
        question = messages[0].content

        context = last_message.content

        # Perform relevance evaluation
        scored_result = chain.invoke({"question": question, "context": context})

        # Extract relevance status
        score = scored_result.binary_score

        if score == "yes":
            print("==== [DECISION: DOCS RELEVANT] ====")
            return "generate_answer_node"
        else:
            print("==== [DECISION: DOCS NOT RELEVANT] ====")
            return "rewrite_question_node"

    return _grade_documents_node
