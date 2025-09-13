import json
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage


class SearchNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, search_config) -> None:
        self.search_tools = TavilySearch(max_results=search_config["num_search_results"])

    def __call__(self, inputs: dict):

        print("Tool Calling")
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            if tool_call["name"] == "tavily_search":
                tool_result = self.search_tools.invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"messages": outputs}
