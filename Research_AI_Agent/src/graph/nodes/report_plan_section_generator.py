from typing import Dict, List
import logging

from ..promps import generate_report_plan_sections_prompt_creator, REPORT_STRUCTURE
from ..helper import load_model
from ..structures import Sections, Section
from ..state import ReportState

logger = logging.getLogger(__name__)


class ReportPlanSectionGenerator:
    """
    A class responsible for generating report plan sections based on a given topic and search context.
    """

    def __init__(
            self,
            model_config: Dict[str, str]
    ):
        """
        Initializes the ReportPlanSectionGenerator with the given model configuration.

        Args:
            model_config: Configuration for the language model.
        """
        self.model_config = model_config
        self.chain = self._initialize_chain()

    def _initialize_chain(
            self
    ):
        """
        Initializes the chain for generating report plan sections.

        Returns:
            The configured chain for section generation.
        """
        logger.info("Initializing the report plan section generation chain.")
        prompt_template = generate_report_plan_sections_prompt_creator(
            report_organization=REPORT_STRUCTURE
        )
        llm = load_model(self.model_config)
        structured_llm = llm.with_structured_output(Sections)
        return prompt_template | structured_llm

    def generate_sections(
            self,
            topic: str,
            search_context: str
    ) -> Sections:
        """
        Generates report plan sections for the given topic and search context.

        Args:
            topic: The topic for which report plan sections are generated.
            search_context: The search context related to the topic.

        Returns:
            The generated report plan sections.
        """
        logger.info("Generating report plan sections.")
        report_sections = self.chain.invoke({"topic": topic, "search_context": search_context})
        return report_sections


class GenerateReportPlanSectionNode:
    """
    A class responsible for generating report plan sections as part of a node in a graph-based workflow.
    """

    def __init__(
            self,
            report_plan_section_generator: ReportPlanSectionGenerator
    ):
        """
        Initializes the GenerateReportPlanSectionNode with a ReportPlanSectionGenerator.

        Args:
            report_plan_section_generator: An instance of ReportPlanSectionGenerator.
        """
        self.report_plan_section_generator = report_plan_section_generator

    def __call__(
            self,
            state: ReportState
    ) -> Dict[str,  List[Section]]:
        """
        Generates report plan sections for the topic and search context in the given state.

        Args:
            state: The state containing the topic and search context for which sections are generated.

        Returns:
            A dictionary containing the generated report plan sections.
        """
        logger.info(" --- Generate Report Plan Sections Node --- ")
        topic = state["topic"]
        search_context = state["search_context"]
        result = self.report_plan_section_generator.generate_sections(topic, search_context)
        logger.debug({"message": "report_plan_sections have been generated",
                      "topic": topic,
                      "search_context": search_context,
                      "report_plan_sections": result})
        return {"sections": result.sections}


def get_generate_report_plan_sections_node(
        model_config: Dict[str, str]
):
    """
    Creates and returns an instance of GenerateReportPlanSectionNode.

    Args:
        model_config: Configuration for the language model.

    Returns:
        An instance of GenerateReportPlanSectionNode.
    """
    logger.info("Creating the `generate report plan sections node`.")
    report_plan_section_generator = ReportPlanSectionGenerator(model_config)
    generate_report_plan_sections_node = GenerateReportPlanSectionNode(
        report_plan_section_generator=report_plan_section_generator
    )
    logger.info("The `generate report plan sections node` has been created.")
    return generate_report_plan_sections_node
