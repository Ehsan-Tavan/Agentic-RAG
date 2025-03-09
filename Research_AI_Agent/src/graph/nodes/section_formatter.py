from typing import Dict, Callable
import logging

from ..structures import Section
from ..state import ReportState

logger = logging.getLogger(__name__)


def format_sections(
        sections: list[Section]
) -> str:
    """
    Formats a list of completed sections into a structured string.

    Args:
        sections: A list of completed sections to format.

    Returns:
        A formatted string representing t   he sections.
    """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
                            {'=' * 60}
                            Section {idx}: {section.name}
                            {'=' * 60}
                            Description:
                            {section.description}
                            Requires Research:
                            {section.research}

                            Content:
                            {section.content if section.content else '[Not yet written]'}

                            """
    return formatted_str


class SectionFormatterNode:
    def __init__(
            self,
            format_sections_fnc: Callable[[list[Section]], str]
    ) -> None:
        self.format_sections_fnc = format_sections_fnc

    def __call__(
            self,
            state: ReportState
    ) -> Dict[str, str]:
        """
        Formats the completed sections in the given state.

        Args:
            state: The state containing the completed sections to format.

        Returns:
            A dictionary containing the formatted sections as a string.
        """
        logger.info("--- Formatting Completed Sections ---")

        # List of completed sections
        completed_sections = state["completed_sections"]

        # Format completed section to str to use as context for final sections
        completed_report_sections = self.format_sections_fnc(completed_sections)

        logger.info("--- Formatting Completed Sections is Done ---")
        logger.debug({"message": "completed sections have been formatted",
                      "formatted_sections": completed_report_sections})

        return {"report_sections_from_research": completed_report_sections}


def get_section_formatter_node() -> SectionFormatterNode:
    """
    Creates and returns an instance of SectionFormatterNode.

    Returns:
        An instance of SectionFormatterNode.
    """
    logger.info("Creating the `section formatter node`.")
    section_formatter_node = SectionFormatterNode(
        format_sections_fnc=format_sections
    )
    logger.info("The `section formatter node` has been created.")
    return section_formatter_node
