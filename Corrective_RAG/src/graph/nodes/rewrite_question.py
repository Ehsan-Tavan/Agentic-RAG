from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Corrective_RAG.src.graph.state import GraphState


class RewriteQueryNode:
    def __init__(self, llm_model):
        self.llm_model = llm_model

        self.chain = self.create_chain()

    def get_prompt(self):
        system = """You a question re-writer that converts an input question to a better version that is optimized 
        for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

        # Create chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        return prompt

    def create_chain(self):
        prompt = self.get_prompt()
        chain = prompt | self.llm_model | StrOutputParser()
        return chain

    def __call__(self, state: GraphState) -> Dict[str, str]:
        print("\n==== [REWRITE QUERY] ====\n")

        return {
            "question":
                self.chain.invoke({"question": state["question"]})
        }


def get_rewrite_query_node(config) -> RewriteQueryNode:
    llm = ChatOpenAI(temperature=config["llm"]["temperature"],
                     model=config["llm"]["model"],
                     api_key=config["llm"]["api_key"],
                     base_url=config["llm"]["base_url"]
                     )
    return RewriteQueryNode(llm_model=llm)
