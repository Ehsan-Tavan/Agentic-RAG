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
