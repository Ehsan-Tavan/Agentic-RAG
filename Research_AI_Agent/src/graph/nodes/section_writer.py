from typing import Dict, List

from Research_AI_Agent.src.graph.promps import generate_section_writer_prompt_creator
from Research_AI_Agent.src.graph.helper import load_model
from Research_AI_Agent.src.graph.state import SectionState
from Research_AI_Agent.src.graph.structures import Section


class SectionWriterNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state: SectionState) -> Dict[str, List[Section]]:
        section = state["section"]
        context = state["source_str"]

        section_content = self.chain.invoke({
            "section_title": section.section_title,
            "section_topic": section.section_topic,
            "context": context
        })

        section.content = section_content.content

        return {
            "completed_sections": [section]
        }


def get_section_writer_chain(model_config: Dict[str, str]):
    prompt_template = generate_section_writer_prompt_creator()
    llm = load_model(model_config)
    chain = prompt_template | llm
    return chain


def get_section_writer_node(model_config: Dict[str, str]):
    section_writer_chain = get_section_writer_chain(model_config)
    section_writer_node = SectionWriterNode(
        chain=section_writer_chain
    )
    return section_writer_node
