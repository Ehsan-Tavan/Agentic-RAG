from typing import Dict

from Research_AI_Agent.src.graph.state import SectionState
from Research_AI_Agent.src.graph.helper import load_model
from Research_AI_Agent.src.graph.promps import final_section_writer_prompt_creator


class FinalSectionWriterNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state: SectionState):
        section = state["section"]
        completed_report_sections = state["report_sections_from_research"]
        print("--- Writing Final Section: " + section.name + " ---")

        section_content = self.chain.invoke({
            "section_title": section.name,
            "section_topic": section.description,
            "context": completed_report_sections,
        })

        section.content = section_content.content

        print("--- Writing Final Section: " + section.name + " Completed ---")
        return {"completed_sections": [section]}


def get_final_section_writer_chain(model_config: Dict[str, str]):
    prompt_template = final_section_writer_prompt_creator()
    llm = load_model(model_config)
    chain = prompt_template | llm
    return chain


def get_final_section_writer_node(model_config: Dict[str, str]) -> FinalSectionWriterNode:
    final_section_writer_chain = get_final_section_writer_chain(model_config)

    final_section_writer_node = FinalSectionWriterNode(
        chain=final_section_writer_chain
    )

    return final_section_writer_node
