from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


from Reflection_Agent.src.graph.state import ReflectionState
from Reflection_Agent.src.graph.structures import DraftCode



class GenerationNode:
    def __init__(self, llm_model):
        self.llm_model = llm_model

        self.chain = self.create_chain()

    @staticmethod
    def _get_prompt():
        system = """You are an expert Python programmer. Write a Python function to solve the following request. 
        Provide a simple, clear implementation and an explanation."""

        # Create chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Request: {user_request}"),
            ]
        )
        return prompt

    def create_chain(self):
        prompt = self._get_prompt()
        chain = prompt | self.llm_model
        return chain

    def __call__(self, state: ReflectionState) -> Dict[str, str]:
        print("--- 1. Generating Initial Draft ---")
        return {
            "draft":
                self.chain.invoke(
                    {"user_request": state["user_request"]}
                )
        }


def get_generation_node(config) -> GenerationNode:
    llm = ChatOpenAI(temperature=config["temperature"],
                     model=config["model"],
                     api_key=config["api_key"],
                     base_url=config["base_url"]
                     )

    llm = llm.with_structured_output(DraftCode)
    return GenerationNode(llm_model=llm)