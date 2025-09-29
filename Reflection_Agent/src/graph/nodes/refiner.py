from typing import Dict
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from Reflection_Agent.src.graph.state import ReflectionState
from Reflection_Agent.src.graph.structures import RefinedCode

class RefinerNode:
    def __init__(self, llm_model):
        self.llm_model = llm_model

        self.chain = self.create_chain()


    @staticmethod
    def _get_prompt():
        system = """You are an expert Python programmer tasked with refining a piece of code based on a critique.
        
        Your goal is to rewrite the original code, implementing all the suggested improvements from the critique.
        """

        # Create chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "**Original Code:**\n```python\n{draft_code}\n```\n\n"
                          "**Critique and Suggestions:**\n{critique_suggestions}\n"
                          "Please provide the final, refined code and a summary of the changes you made."),
            ]
        )
        return prompt

    def create_chain(self):
        prompt = self._get_prompt()
        chain = prompt | self.llm_model
        return chain

    def __call__(self, state: ReflectionState) -> Dict[str, str]:
        print("--- 3. Refining Code ---")
        return {
            "refined_code":
                self.chain.invoke(
                    {"draft_code": state["draft"].code,
                     "critique_suggestions": json.dumps(state['critique'].model_dump(), indent=2)}
                )
        }

def get_refiner_node(config) -> RefinerNode:
    llm = ChatOpenAI(temperature=config["temperature"],
                     model=config["model"],
                     api_key=config["api_key"],
                     base_url=config["base_url"]
                     )

    llm = llm.with_structured_output(RefinedCode)
    return RefinerNode(llm_model=llm)