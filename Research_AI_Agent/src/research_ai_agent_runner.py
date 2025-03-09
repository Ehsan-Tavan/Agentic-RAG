import os
import logging
import argparse
import yaml
import asyncio
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from graph.research_ai_agent import create_reporter_agent
from utils import create_logger


async def call_planner_agent(agent,
                             topic: str,
                             output_path: str,
                             config: dict = {"recursion_limit": 50},
                             verbose: bool = False):
    """
        Calls the planner agent to generate a report and saves it to a markdown file.

        Args:
            agent: The reporter agent instance.
            topic: The research topic.
            output_path: Path where the markdown report will be saved.
            config (dict): Configuration for the agent's streaming process.
            verbose: If True, prints intermediate steps.
        """
    console = Console()
    events = agent.astream(
        {"topic": topic},
        config,
        stream_mode="values",
    )

    async for event in events:
        for k, v in event.items():
            if verbose:
                if k != "__end__":
                    console.print(RichMarkdown(f"{repr(k)} -> {repr(v)}"))
            if k == "final_report":
                print("=" * 50)
                print("Final Report:")
                md = RichMarkdown(v)
                console.print(md)
                try:
                    # Ensure the output directory exists
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(v)
                        logger.info(f"Report successfully saved to {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save report to {output_path}: {e}")


async def main(agent, topic: str, output_path: str):
    """
    Runs the main process to generate a report for the given topic.

    Args:
        agent: The reporter agent instance.
        topic (str): The research topic.
        output_path (str): Path where the markdown report will be saved.
    """
    await call_planner_agent(agent=agent, topic=topic, output_path=output_path)


if __name__ == "__main__":
    # Initialize logging
    create_logger()
    logger = logging.getLogger(__name__)

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Deep Research AI Agent"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        type=str,
        help="Config file path (default: None)"
    )
    parser.add_argument(
        "-t", "--topic",
        default="Detailed report on how to build Agentic AI systems, design patterns and current frameworks",
        type=str,
        help="Config file path (default: None)"
    )
    parser.add_argument(
        "-o", "--output",
        default="report.md",
        type=str,
        help="Output markdown file path (default: report.md)"
    )
    args = parser.parse_args()

    # Validate required arguments
    if args.config is None:
        raise ValueError("The config argument should be set!")

    # Load configuration
    config = yaml.safe_load(open(args.config))
    logger.debug({"config": config})

    # Set environment variables for API keys
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", config["model_config"].get("api_key", None))
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY",
                                             config["web_search_config"].get("tavily_api_key", None))

    # Initialize search wrapper and agent
    tavily_search = TavilySearchAPIWrapper()
    bot = create_reporter_agent(config=config, tavily_search=tavily_search)

    # Run the main process
    asyncio.run(main(agent=bot, topic=args.topic, output_path=args.output))
