from langchain_core.prompts import ChatPromptTemplate

REPORT_ORGANIZATION = """You are an expert technical report writer, helping to plan a report.

The report will be focused on the following topic:
{topic}

The report structure will follow these guidelines:
{report_organization}

Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for 
planning the report sections.

The query should:
1. Be related to the topic
2. Help satisfy the requirements specified in the report organization

Make the query specific enough to find high-quality, relevant sources while covering the depth and breadth needed 
for the report structure.
"""


def generate_report_plan_prompt_creator(
        number_of_queries: int,
        report_organization: str
) -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            ("system", "You are an expert technical report writer, helping to plan a comprehensive report."
                       f"Your goal is to generate {number_of_queries} search queries that will help gather "
                       "comprehensive information for planning the report sections.\n\n"
                       "The report structure will follow these guidelines:\n"
                       f"{report_organization}\n\n"
                       "**Guidelines for Generating Queries:**\n"
                       "1. Ensure each query is directly related to the topic.\n"
                       "2. Tailor the queries to satisfy the requirements outlined in the report structure.\n"
                       "3. Make the query specific enough to find high-quality, relevant sources while covering the "
                       "depth and breadth needed for the report structure.\n"
             ),
            ("user",
             "Report Topic:\n{topic}\n\n"
             "Generate a list of search queries that will help gather information for a detailed and well-structured "
             "report on this topic.")
        ]
    )


def generate_report_plan_sections_prompt_creator(report_organization) -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            ("system",
             "You are an expert technical report writer, helping to plan a report.\n\n"
             "Your goal is to generate the outline of the sections of the report based on the provided details.\n\n"
             f"The report should follow this report structure:\n{report_organization}\n\n"
             "**Guidelines for Generating Sections:**\n"
             "- Each section should include:\n"
             "  - **Name**: Name for this section of the report.\n"
             "  - **Description**: Brief overview of the main topics and concepts to be covered in this section.\n"
             "  - **Research**: Indicate whether additional web search is needed for this section or not.\n"
             "  - **Content**: The content of the section, which you will leave blank for now.\n\n"
             "Consider which sections require web search.\n"
             "For example, introduction and conclusion will not require research because they will distill "
             "information from other parts of the report.\n\n"
             ),
            ("user",
             "Report Topic:\n{topic}\n\n"
             "Reflect on this additional context information from web searches:\n{search_context}\n\n"
             "Generate a structured outline for the report sections, following the given report structure and using "
             "the web search context to enhance the section planning.")
        ]
    )


def generate_section_research_queries_prompt_creator(number_of_queries: int) -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            ("system",
             "You are an expert in technical research, skilled in crafting precise and targeted search queries to "
             "gather relevant information.\n\n"
             "Your task is to generate targeted web search queries that will help collect comprehensive information "
             "for writing a technical report section.\n\n"
             "When generating search queries, ensure that they:\n"
             "  1. Cover different aspects of the section topic (e.g., technical details, real-world applications, "
             "comparisons, challenges).\n"
             "  2. Include specific technical terms relevant to the topic.\n"
             "  3. Target recent information by including year markers where relevant (e.g., '2024')"
             "  4. Look for comparisons or differentiators from similar technologies/approaches.\n"
             "  5. Search for both official documentation and practical implementation examples.\n\n"
             f"Generate {number_of_queries} well-formed search queries based on the section description.\n\n"
             "The search queries should be:\n"
             "- Specific enough to avoid generic results.\n"
             "- Technical enough to capture detailed implementation information\n"
             "- Diverse enough to cover all aspects of the section plan\n"
             "- Focused on authoritative sources (documentation, technical blogs, academic papers)\n"
             ),
            ("user",
             "Section Topic: {section_topic}\n\n"
             "Generate a diverse and targeted set of search queries for this section.")
        ]
    )
