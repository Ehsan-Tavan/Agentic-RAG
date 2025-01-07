from ..state import State


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
