from typing import Dict, List, Callable
import logging

from ..state import ReportState
from ..structures import Section

logger = logging.getLogger(__name__)


def compile_final_report(
        sections: List[Section],
        completed_sections: Dict[str, str]
) -> str:
    """
    Compiles the final report by updating sections with completed content and formatting the output.

    Args:
        sections: A list of sections to include in the final report.
        completed_sections: A dictionary mapping section names to their completed content.

    Returns:
        The compiled and formatted final report as a string.
    """
    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    # Escape unescaped $ symbols to display properly in Markdown
    formatted_sections = all_sections.replace(
        "\\$", "TEMP_PLACEHOLDER"
    )  # Temporarily mark already escaped $
    formatted_sections = formatted_sections.replace(
        "$", "\\$"
    )  # Escape all $
    formatted_sections = formatted_sections.replace(
        "TEMP_PLACEHOLDER", "\\$"
    )  # Restore originally escaped $

    return formatted_sections


class FinalReportCompilerNode:
    """
    A class responsible for compiling the final report as part of a node in a graph-based workflow.
    """

    def __init__(
            self,
            final_report_compiler: Callable[[List[Section], Dict[str, str]], str]
    ):
        """
        Initializes the FinalReportCompilerNode with a FinalReportCompiler.

        Args:
            final_report_compiler: An instance of FinalReportCompiler.
        """
        self.final_report_compiler = final_report_compiler

    def __call__(
            self,
            state: ReportState
    ) -> Dict[str, str]:
        """
        Compiles the final report from the given state.

        Args:
            state: The state containing the sections and completed sections.

        Returns:
            A dictionary containing the compiled final report.
        """
        logger.info("--- Compiling Final Report ---")
        sections = state["sections"]
        completed_sections = {s.name: s.content for s in state["completed_sections"]}

        final_report = self.final_report_compiler(sections, completed_sections)

        logger.info("--- Compiling Final Report Done ---")
        logger.debug({"message": "final report has been compiled",
                      "final_report": final_report})
        return {"final_report": final_report}


def get_final_report_compiler_node() -> FinalReportCompilerNode:
    """
    Creates and returns an instance of FinalReportCompilerNode.

    Returns:
        An instance of FinalReportCompilerNode.
    """
    logger.info("Creating the `final report compiler node`.")
    final_report_compiler_node = FinalReportCompilerNode(
        final_report_compiler=compile_final_report
    )
    logger.info("The `final report compiler node` has been created.")
    return final_report_compiler_node
