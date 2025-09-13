from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from Corrective_RAG.src.graph.state import GraphState


class GenerationNode:
    def __init__(self, llm_model):
        self.llm_model = llm_model

        self.chain = self.create_chain()

    def get_prompt(self):
        system = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
        to answer the question. If you don't know the answer, just say that you don't know. Use three sentences 
        maximum and keep the answer concise."""

        # Create chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Question: {question}\nContext: {context}\nAnswer:"),
            ]
        )
        return prompt

    def create_chain(self):
        prompt = self.get_prompt()
        chain = prompt | self.llm_model | StrOutputParser()
        return chain

    def __call__(self, state: GraphState) -> Dict[str, str]:
        print("\n==== GENERATE ====\n")
        return {
            "generation":
                self.chain.invoke({
                    "context": [doc.page_content for doc in state["documents"]],
                    "question": state["question"]})
        }


def get_generation_node(config) -> GenerationNode:
    llm = ChatOpenAI(temperature=config["llm"]["temperature"],
                     model=config["llm"]["model"],
                     api_key=config["llm"]["api_key"],
                     base_url=config["llm"]["base_url"]
                     )
    return GenerationNode(llm_model=llm)
