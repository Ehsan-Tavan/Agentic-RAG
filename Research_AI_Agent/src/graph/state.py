from typing import TypedDict, Annotated, List
import operator
from Research_AI_Agent.src.graph.structures import Section, SearchQuery


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


class SectionState(TypedDict):
    section: Section  # Report section
    research_queries: list[SearchQuery]  # List of search queries
    search_context: str  # String of formatted source content from web search
    report_sections_from_research: str  # String of any completed sections from research to write final sections
    completed_sections: list[Section]  # Final key we duplicate in outer state for Send() API


class SectionOutputState(TypedDict):
    completed_sections: list[Section]  # Final key we duplicate in outer state for Send() API
