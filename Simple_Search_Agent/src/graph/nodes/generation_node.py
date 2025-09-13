from Simple_Search_Agent.src.graph import State
from langchain_openai import ChatOpenAI


class GenerationNode:
    def __init__(self, llm_configs, tools):
        model = ChatOpenAI(temperature=llm_configs["temperature"],
                           model=llm_configs["model"],
                           streaming=True)

        self.model = model.bind_tools(tools)

    def __call__(self, state: State):
        return {"messages": [self.model.invoke(state["messages"])]}
