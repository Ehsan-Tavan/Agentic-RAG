import os
import logging
import argparse
import yaml
import asyncio
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from Research_AI_Agent.src.graph.research_ai_agent import create_reporter_agent
from Research_AI_Agent.src.utils import create_logger


async def call_planner_agent(agent, prompt, config={"recursion_limit": 50}, verbose=False):
    console = Console()
    events = agent.astream(
        {'topic': prompt},
        config,
        stream_mode="values",
    )

    async for event in events:
        for k, v in event.items():
            if verbose:
                if k != "__end__":
                    # print(k)
                    # print(v)
                    # print("################")
                    # display(RichMarkdown(repr(k) + ' -> ' + repr(v)))
                    console.print(RichMarkdown(repr(k) + ' -> ' + repr(v)))
            if k == 'final_report':
                print('=' * 50)
                print('Final Report:')
                md = RichMarkdown(v)
                console.print(md)
                # print(v)


async def main(agent):
    topic = "Detailed report on how to build Agentic AI systems, design patterns and current frameworks"
    await call_planner_agent(agent=agent,
                             prompt=topic)
    # docs = await run_search_queries(['langgarph'], include_raw_content=True)
    #
    # output = format_search_query_results(docs, max_tokens=500, include_raw_content=True)
    # print(output)


if __name__ == "__main__":
    create_logger()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Deep Research AI Agent"
    )
    parser.add_argument("-c", "--config", default=None, type=str,
                        help="Config file path (default: None)")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("The config argument should be set!")

    config = yaml.safe_load(open(args.config))
    logger.debug({"config": config})

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", config["model_config"].get("api_key", None))
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", config.get("tavily_api_key", None))
    tavily_search = TavilySearchAPIWrapper()

    bot = create_reporter_agent(config=config, tavily_search=tavily_search)

    asyncio.run(main(agent=bot))
