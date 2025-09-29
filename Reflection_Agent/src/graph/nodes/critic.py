from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


from Reflection_Agent.src.graph.state import ReflectionState
from Reflection_Agent.src.graph.structures import Critique


class CriticNode:
    def __init__(self, llm_model):
        self.llm_model = llm_model

        self.chain = self.create_chain()


    @staticmethod
    def _get_prompt():
        system = """You are an expert code reviewer and senior Python developer. Your task is to perform a thorough critique of the following code.
        Analyze the code for:
        1.  **Bugs and Errors:** Are there any potential runtime errors, logical flaws, or edge cases that are not handled?
        2.  **Efficiency and Best Practices:** Is this the most efficient way to solve the problem? Does it follow standard Python conventions (PEP 8)?
        
        Provide a structured critique with specific, actionable suggestions."""

        # Create chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Code to Review:\n```python\n{code_to_critique}\n```"),
            ]
        )
        return prompt

    def create_chain(self):
        prompt = self._get_prompt()
        chain = prompt | self.llm_model
        return chain

    def __call__(self, state: ReflectionState) -> Dict[str, str]:
        print("--- 2. Critiquing Draft ---")
        return {
            "critique":
                self.chain.invoke(
                    {"code_to_critique": state["draft"].code}
                )
        }



def get_critic_node(config) -> CriticNode:
    llm = ChatOpenAI(temperature=config["temperature"],
                     model=config["model"],
                     api_key=config["api_key"],
                     base_url=config["base_url"]
                     )

    llm = llm.with_structured_output(Critique)
    return CriticNode(llm_model=llm)
