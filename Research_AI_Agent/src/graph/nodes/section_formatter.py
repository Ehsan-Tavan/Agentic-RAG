from Research_AI_Agent.src.graph.structures import Section
from Research_AI_Agent.src.graph.state import ReportState


def format_sections(sections: list[Section]) -> str:
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


def build_completed_sections_string(state: ReportState):
    print('--- Formatting Completed Sections ---')

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    print('--- Formatting Completed Sections is Done ---')

    return {"report_sections_from_research": completed_report_sections}
