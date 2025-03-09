from langchain_core.prompts import ChatPromptTemplate

REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided topic
                      and building a comprehensive report in markdown using the following format:

                      1. Introduction (no web search needed)
                            - Brief overview of the topic area

                      2. Main Body Sections:
                            - Each section should focus on a sub-topic of the user-provided topic
                            - Include any key concepts and definitions
                            - Provide real-world examples or case studies where applicable

                      3. Conclusion (no web search needed)
                            - Aim for 1 structural element (either a list of table) that distills the main body sections
                            - Provide a concise summary of the report

                      When generating the final response in markdown, if there are special characters in the text,
                      such as the dollar symbol, ensure they are escaped properly for correct rendering e.g $25.5 should become \$25.5
                  """


def generate_topic_research_queries_prompt_creator(
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


def generate_section_writer_prompt_creator() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            ("system",
             "You are an expert technical writer crafting one specific section of a technical report.\n\n"
             "Your task is to write a concise, accurate, and well-structured section based on the given topic.\n\n"
             "Follow these strict guidelines:\n"
             "1. **Technical Accuracy:**\n"
             "   - Include specific version numbers where applicable.\n"
             "   - Reference concrete metrics, benchmarks, and official documentation.\n"
             "   - Use precise technical terminology.\n\n"
             "2. **Length and Style:**\n"
             "   - Strictly **150-200 words**.\n"
             "   - No marketing language—focus purely on technical details.\n"
             "   - Use clear, simple language without unnecessary complexity.\n"
             "   - Start with the **most important insight in bold**.\n"
             "   - Keep paragraphs short (2-3 sentences max).\n\n"
             "3. **Structure:**\n"
             "   - Use `##` for the section title (Markdown format).\n"
             "   - Use **only ONE** structural element **IF** it clarifies the point:\n"
             "     - Either a small comparison table (Markdown format) OR a short bullet list (3-5 items).\n"
             "   - End with `### Sources`, listing references in the format:\n"
             "     `- Title : URL`\n\n"
             "4. **Writing Approach:**\n"
             "   - Provide at least **one** specific example or case study if available.\n"
             "   - Use **concrete details** over general statements.\n"
             "   - No preamble—start directly with the section content.\n"
             "   - Every word should be **precise and meaningful**.\n\n"
             "5. **Formatting & Quality Checks:**\n"
             "   - Ensure the text is **Markdown-formatted**.\n"
             "   - The section **must** be between **150-200 words** (excluding sources).\n"
             "   - If using special characters (e.g., `$` for currency), escape them properly (`\\$`).\n\n"
             "Use the provided source material to ensure accuracy."
             ),
            ("user",
             "Title: {section_title}\n\n"
             "Topic: {section_topic}\n\n"
             "Source Material:\n"
             "{context}\n\n"
             "Write the section following all given instructions.")
        ]
    )


def final_section_writer_prompt_creator() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            ("system",
             "You are an expert technical writer crafting a section that synthesizes information from the rest of "
             "the report.\n\n"
             "Your task is to create a concise, well-structured section based on the provided context.\n\n"
             "Follow these strict guidelines:\n"
             "1. **Section-Specific Approach**:\n"
             "   - **For Introduction**:\n"
             "     - Use `#` for the report title (Markdown format).\n"
             "     - Limit to **50-100 words**.\n"
             "     - Use simple and clear language.\n"
             "     - Focus on the core motivation for the report in **1-2 paragraphs**.\n"
             "     - Maintain a clear narrative arc.\n"
             "     - **Do NOT** include any structural elements (no lists, no tables).\n"
             "     - **No sources section** needed.\n"
             "   - **For Conclusion/Summary**:\n"
             "     - Use `##` for the section title (Markdown format).\n"
             "     - Limit to **100-150 words**.\n"
             "     - For **comparative reports**:\n"
             "       * Include a focused **comparison table** using Markdown table syntax.\n"
             "       * The table should distill insights from the report.\n"
             "     - For **non-comparative reports**:\n"
             "       * Use **ONLY ONE** structural element if it helps distill the points:\n"
             "         - A focused table (Markdown format) OR a short list (3-5 items).\n"
             "       * End with specific **next steps** or **implications**.\n"
             "     - **No sources section** needed.\n"
             "2. **Writing Approach**:\n"
             "   - Provide **concrete details** over general statements.\n"
             "   - **Every word** should count and focus on your most important point.\n"
             "3. **Formatting & Quality Checks**:\n"
             "   - Ensure the section is **Markdown-formatted**.\n"
             "   - The introduction must be **50-100 words** with no structural elements.\n"
             "   - The conclusion must be **100-150 words** with **only one** structural element.\n"
             "   - Escape special characters properly (e.g., \$25.5 becomes `\\$25.5`).\n"
             "   - Do not include word count or any preamble in the response.\n"
             ""),
            ("user",
             "Title for the section: {section_title}\n\n"
             "Topic for this section: {section_topic}\n\n"
             "Available report content of already completed sections:\n"
             "{context}\n\n"
             "Write the section following all given instructions.")
        ]
    )
