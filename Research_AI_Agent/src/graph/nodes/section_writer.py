from typing import Dict, List
import logging

from Research_AI_Agent.src.graph.promps import generate_section_writer_prompt_creator
from Research_AI_Agent.src.graph.helper import load_model
from Research_AI_Agent.src.graph.state import SectionState
from Research_AI_Agent.src.graph.structures import Section

logger = logging.getLogger(__name__)


class SectionWriter:
    """
    A class responsible for writing content for a specific section.
    """

    def __init__(
            self,
            model_config: Dict[str, str]
    ):
        """
        Initializes the SectionWriter with the given model configuration.

        Args:
            model_config: Configuration for the language model.
        """
        self.model_config = model_config
        self.chain = self._initialize_chain()

    def _initialize_chain(
            self
    ):
        """
        Initializes the chain for writing section content.

        Returns:
            The configured chain for section writing.
        """
        logger.info("Initializing the section writer chain.")
        prompt_template = generate_section_writer_prompt_creator()
        llm = load_model(self.model_config)
        return prompt_template | llm

    def write_section(
            self,
            section_title: str,
            section_topic: str,
            context: str
    ) -> str:
        """
        Writes content for a section based on the provided title, topic, and context.

        Args:
            section_title: The title of the section.
            section_topic: The topic of the section.
            context: The context or background information for the section.

        Returns:
            The generated content for the section.
        """
        logger.info(f"Writing content for section: {section_title}")
        section_content = self.chain.invoke({
            "section_title": section_title,
            "section_topic": section_topic,
            "context": context
        })
        return section_content.content


class SectionWriterNode:
    """
    A class responsible for writing section content as part of a node in a graph-based workflow.
    """

    def __init__(
            self,
            section_writer: SectionWriter
    ):
        """
        Initializes the SectionWriterNode with a SectionWriter.

        Args:
            section_writer: An instance of SectionWriter.
        """
        self.section_writer = section_writer

    def __call__(
            self,
            state: SectionState
    ) -> Dict[str, List[Section]]:
        """
        Writes content for the section in the given state.

        Args:
            state: The state containing the section and context for which content is written.

        Returns:
            A dictionary containing the completed section with its generated content.
        """
        logger.info(" --- SectionWriterNode --- ")
        section = state["section"]
        context = state["search_context"]

        section_content = self.section_writer.write_section(
            section_title=section.name,
            section_topic=section.description,
            context=context
        )
        section.content = section_content

        logger.debug({"message": "section content has been generated",
                      "section_name": section.name,
                      "section_content": section_content})
        return {"completed_sections": [section]}


def get_section_writer_node(
        model_config: Dict[str, str]
):
    """
    Creates and returns an instance of SectionWriterNode.

    Args:
        model_config: Configuration for the language model.

    Returns:
        An instance of SectionWriterNode.
    """
    logger.info("Creating the `section writer node`.")
    section_writer = SectionWriter(model_config)
    section_writer_node = SectionWriterNode(
        section_writer=section_writer
    )
    logger.info("The `section writer node` has been created.")
    return section_writer_node
