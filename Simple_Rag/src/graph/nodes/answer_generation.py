from typing import Callable
from Simple_Rag.src.graph import State
from Simple_Rag.src.llm_model import get_llm_chain


class AnswerGenerationNode:
    def __init__(
            self,
            answer_generation_chain
    ):
        """
        Node responsible for generating answers using a language model chain.

        Args:
            A callable object (like an LLM chain) that generates answers
            based on the provided context and question.
        """
        self.answer_generation_chain = answer_generation_chain

    def __call__(
            self,
            state: State
    ) -> dict:
        """
        Process the current state to generate an answer.

        Args:
            state: The current state containing the question and context.

        Returns:
            A dictionary with the key "answer" containing the generated answer.
        """
        return {
            "answer": self.answer_generation_chain.invoke(
                {
                    "context": state["context"],
                    "question": state["question"],
                }
            )
        }


def get_answer_generation_node(llm_config: dict) -> AnswerGenerationNode:
    """
    Create an instance of AnswerGenerationNode using the specified LLM configuration.

    Args:
        llm_config: Configuration details for initializing the LLM chain.

    Returns:
        An instance of AnswerGenerationNode configured with the LLM chain.
    """
    llm_chain = get_llm_chain(llm_config)
    return AnswerGenerationNode(llm_chain)
