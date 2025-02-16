from typing import TypedDict, Annotated, List
import operator
from Research_AI_Agent.src.graph.structures import Section, Queries


class ReportStateInput(TypedDict):
    topic: str  # Report topic


class ReportStateOutput(TypedDict):
    final_report: str  # Final report


class ReportState(TypedDict):
    topic: str  # Report topic
    research_queries: List[str]
    search_context: str
    sections: List[Section]  # List of report sections
    completed_sections: Annotated[list, operator.add]  # Send() API
    report_sections_from_research: str  # String of any completed sections from research to write final sections
    final_report: str  # Final report
