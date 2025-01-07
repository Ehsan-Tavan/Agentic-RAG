from Simple_Rag.src.graph import State
from Simple_Rag.src.llm_model import get_llm_chain


class AnswerGenerationNode:
    def __init__(self, answer_generation_chain):
        self.answer_generation_chain = answer_generation_chain

    def __call__(self, state: State):
        return {
            "answer": self.answer_generation_chain.invoke(
                {
                    "context": state["context"],
                    "question": state["question"],
                }
            )
        }


def get_answer_generation_node(llm_config):
    llm_chain = get_llm_chain(llm_config)
    return AnswerGenerationNode(llm_chain)
