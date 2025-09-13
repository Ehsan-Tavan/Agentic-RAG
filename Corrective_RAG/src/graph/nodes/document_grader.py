from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from Corrective_RAG.src.graph.state import GraphState



class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved document."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class DocumentGraderNode:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.chain = self.create_chain()


    def get_prompt(self):
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        # Create chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        return prompt

    def create_model(self):
        return self.llm_model.with_structured_output(GradeDocuments)


    def create_chain(self):
        prompt  = self.get_prompt()
        structured_llm_grader = self.create_model()
        chain = prompt | structured_llm_grader
        return chain


    def __call__(self, state: GraphState) -> Dict[str, str]:
        print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
        filtered_docs = []
        relevant_doc_count = 0

        for d in state["documents"]:
            score = self.chain.invoke(
                {"question": state["question"], "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("==== [GRADE: DOCUMENT RELEVANT] ====")
                # Add relevant documents to filtered_docs.
                filtered_docs.append(d)
                relevant_doc_count += 1
            else:
                print("==== [GRADE: DOCUMENT NOT RELEVANT] ====")
                continue
        web_search = "Yes" if relevant_doc_count == 0 else "No"

        return {"documents": filtered_docs, "web_search": web_search}



def get_document_grader_node(config: dict):
    llm = ChatOpenAI(temperature=config["llm"]["temperature"],
                       model=config["llm"]["model"],
                       api_key=config["llm"]["api_key"],
                       base_url=config["llm"]["base_url"]
                       )
    return DocumentGraderNode(llm_model=llm)