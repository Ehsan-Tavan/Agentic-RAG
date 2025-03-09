from typing import Dict, List
import logging

from Research_AI_Agent.src.graph.state import SectionState, Section
from Research_AI_Agent.src.graph.helper import load_model
from Research_AI_Agent.src.graph.promps import final_section_writer_prompt_creator

logger = logging.getLogger(__name__)


class FinalSectionWriter:
    """
    A class responsible for writing the final content for a specific section.
    """

    def __init__(
            self,
            model_config: Dict[str, str]
    ):
        """
        Initializes the FinalSectionWriter with the given model configuration.

        Args:
            model_config: Configuration for the language model.
        """
        self.model_config = model_config
        self.chain = self._initialize_chain()

    def _initialize_chain(
            self
    ):
        """
        Initializes the chain for writing final section content.

        Returns:
            The configured chain for final section writing.
        """
        logger.info("Initializing the final section writer chain.")
        prompt_template = final_section_writer_prompt_creator()
        llm = load_model(self.model_config)
        return prompt_template | llm

    def write_final_section(
            self,
            section_title: str,
            section_topic: str,
            context: str
    ) -> str:
        """
        Writes the final content for a section based on the provided title, topic, and context.

        Args:
            section_title: The title of the section.
            section_topic: The topic of the section.
            context: The context or background information for the section.

        Returns:
            The generated final content for the section.
        """
        logger.info(f"Writing final content for section: {section_title}")
        section_content = self.chain.invoke({
            "section_title": section_title,
            "section_topic": section_topic,
            "context": context
        })
        return section_content.content


class FinalSectionWriterNode:
    """
    A class responsible for writing final section content as part of a node in a graph-based workflow.
    """

    def __init__(
            self,
            final_section_writer: FinalSectionWriter
    ):
        """
        Initializes the FinalSectionWriterNode with a FinalSectionWriter.

        Args:
            final_section_writer: An instance of FinalSectionWriter.
        """
        self.final_section_writer = final_section_writer

    def __call__(
            self,
            state: SectionState
    ) -> Dict[str, List[Section]]:
        """
        Writes the final content for the section in the given state.

        Args:
            state: The state containing the section and context for which final content is written.

        Returns:
            A dictionary containing the completed section with its final content.
        """
        logger.info(f"--- Writing Final Section: {state['section'].name} ---")
        section = state["section"]
        completed_report_sections = state["report_sections_from_research"]

        section_content = self.final_section_writer.write_final_section(
            section_title=section.name,
            section_topic=section.description,
            context=completed_report_sections
        )
        section.content = section_content

        logger.info(f"--- Writing Final Section: {section.name} Completed ---")
        logger.debug({"message": "final section content has been generated",
                      "section_name": section.name,
                      "section_content": section_content})
        return {"completed_sections": [section]}


def get_final_section_writer_node(
        model_config: Dict[str, str]
) -> FinalSectionWriterNode:
    """
    Creates and returns an instance of FinalSectionWriterNode.

    Args:
        model_config: Configuration for the language model.

    Returns:
        An instance of FinalSectionWriterNode.
    """
    logger.info("Creating the `final section writer node`.")
    final_section_writer = FinalSectionWriter(model_config)
    final_section_writer_node = FinalSectionWriterNode(
        final_section_writer=final_section_writer
    )
    logger.info("The `final section writer node` has been created.")
    return final_section_writer_node
