from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


def get_llm_model(llm_config: dict) -> ChatOpenAI:
    """
    Initialize an OpenAI language model with the given configuration.

    Args:
        llm_config: Configuration details for the language model, including:
            - "name" (str): The model's name (e.g., "gpt-4").
            - "temperature" (float): Sampling temperature for response variability.
            - "api_key" (str): OpenAI API key for authentication.
            - "base_url" (str): Base URL for API requests.

    Returns:
        An instance of the OpenAI language model.
    """
    return ChatOpenAI(
        model=llm_config["name"],
        temperature=llm_config["temperature"],
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
    )


def get_prompt() -> ChatPromptTemplate:
    """
    Create a chat prompt template with predefined system instructions and a user message format.

    Returns:
        A template for generating prompts for the language model.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful AI assistant. Follow these rules when answering questions:\n"
             "- Base your answers strictly on the given Context and provide as much detail as possible.\n"
             "- Be polite, clear, and precise.\n"
             "- If the answer cannot be derived from the Context, respond with: "
             "'Based on my knowledge, I cannot answer your question.'\n"
             "- Avoid generating harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."),
            ("user",
             "Using the following context, please answer my question:\n\n"
             "## Context:\n{context}\n\n"
             "## Question:\n{question}\n\n"
             "## Answer:")
        ]
    )


def get_llm_chain(llm_config: dict) -> LLMChain:
    """
    Create a chain combining a prompt and a language model for question answering.

    Args:
        llm_config: Configuration details for the language model.

    Returns:
        A chain combining the language model and prompt for question-answering tasks.
    """
    prompt = get_prompt()
    model = get_llm_model(llm_config)

    llm_chain = prompt | model  # Using the pipeline operator to combine the prompt with the model.
    return llm_chain
