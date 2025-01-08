from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_chunker(chunker_config: dict):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunker_config["chunk_size"],
        chunk_overlap=chunker_config["chunk_overlap"],
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter
