from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def get_llm_model(llm_config):
    return ChatOpenAI(
        model=llm_config["name"],
        temperature=llm_config["temperature"],
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
    )


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful AI assistant. Follow these rules when answering questions:\n"
             "- Base your answers strictly on the given Context and provide as much detail as possible.\n"
             "- Be polite, clear, and precise.\n"
             "- If the answer cannot be derived from the Context, respond with: 'Based on my knowledge, I cannot "
             "answer your question.'\n"
             "- Avoid generating harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."),

            ("user",
             "Using the following context, please answer my question:\n\n"
             "## Context:\n{context}\n\n"
             "## Question:\n{question}\n\n"
             "## Answer:")
        ]
    )


def get_llm_chain(llm_config):
    prompt = get_prompt()
    model = get_llm_model(llm_config)

    llm_chain = prompt | model
    return llm_chain

