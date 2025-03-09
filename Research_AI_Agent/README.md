# Deep Research AI Agent
Welcome to the Deep Research AI Agent, an autonomous system designed to 
conduct in-depth research on any topic you provide. Powered by language 
models and web search capabilities, this agent generates research queries, 
gathers information from the web, and compiles it into comprehensive 
reports—perfect for researchers, students, and professionals looking 
to streamline their research process.


## Features
- **Dynamic Query Generation:** Automatically creates research queries tailored to your topic.
- **Web Search Integration:** Uses the Tavily API to fetch up-to-date information from the web.
- **Section-Based Report Writing:** Breaks down the report into sections, handling research-heavy and non-research sections separately.
- **Parallel Processing:** Speeds up research by processing multiple sections simultaneously.
- **Customizable Workflow:** Adjust settings like the number of queries or model parameters to suit your needs.

## How It Works

The Deep Research AI Agent is built using [LangGraph](https://langchain-ai.github.io/langgraph/), 
a library that enables complex, stateful  workflows with language models. The agent operates as 
a multi-step process, orchestrated through a state graph with specialized nodes:

1. **Topic Research Queries:** Generates a set of queries based on your input topic.
2. **Web Search:** Performs searches using the Tavily API to collect relevant data.
3. **Report Planning:** Organizes the report into sections, identifying which require research.
4. **Section Building:** A sub-agent handles research-heavy sections in parallel, generating queries, 
searching, and writing content.
5. **Section Formatting:** Polishes the completed sections for consistency.
6. **Final Writing:** Writes non-research sections using the research findings.
7. **Report Compilation:** Assembles all sections into a final, cohesive report.

For a visual overview of the workflow, check out [research_ai_agent.png](images/research_ai_agent.png)
and [section_builder_agent.png](images/section_builder_agent.png) 
in the repository.

## Installation

Follow these steps to set up the Deep Research AI Agent on your system:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Ehsan-Tavan/Agentic-RAG.git
    ```

2. **Navigate to the Project Directory:**
    ```bash
    cd deep-research-ai-agent
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up the APIs:**
   - **OpenAI API:**
     - Add the key to your environment variables:
     ```bash
     export OPENAI_API_KEY="your-openai-api-key-here"
     ```

   - **Tavily API:**
     - Sign up at [Tavily](https://tavily.com/) to get an API key.
     - Add the key to your environment variables:
       ```bash
       export TAVILY_API_KEY="your-api-key-here"
       ```
       Alternatively, configure it in the [config.json](configs/config.yaml) file (see Configuration).

## Usage

To start researching a topic and generate a report, run the agent with a simple command:
```bash
python src/research_ai_agent_runner.py --topic "Your Research Topic" --output "path/to/your/report.md"
```

- `--topic`: A required argument where you specify the subject you’d like to explore. 
Just replace "Your Research Topic" with your own idea!

- `--output`: An optional argument to specify the file path where the markdown report will 
be saved. If omitted, it defaults to report.md in the current directory.

### What Happens Next?
When you execute the command, the agent will:

- Generate tailored research queries based on your topic.
- Search the web for the latest, relevant information.
- Compile the findings into a comprehensive report.

The final report will be:
- **Displayed in the console** with clear, formatted output.
- **Saved to the specified markdown file** (e.g., `report.md` or your 
custom path) for future reference.

## Configuration
The Deep Research AI Agent is highly customizable through the [config.json](configs/config.yaml) file, 
which controls both the language model and web search behavior. Below is 
an overview of the available settings, their purposes, and how to adjust 
them to suit your needs.

### File Structure
he configuration is split into two main sections: `model_config` and 
`web_search_config`. You can edit these settings in the `config.yaml` 
file.

### Example Config
Here’s a sample `config.yaml` with default values:
```yaml
model_config:
  # Name of the model being used.
  name: ""

  # Controls the randomness of the model's output.
  temperature: 0

  # API key for authenticating requests to the model's API.
  api_key: ""

  # Base URL for the API endpoint where the model is hosted.
  base_url: ""

web_search_config:
  # API key for the Tavily service, used for web search or data retrieval.
  tavily_api_key: ""

  # Number of search queries to generate for the topic.
  number_of_queries_for_topic: 8

  # Number of search queries to generate for each report section.
  number_of_queries_for_section: 5

  # Number of search results to return for each query.
  num_results: 6

  # Whether to include raw content (e.g., full HTML or text) in the search results.
  include_raw_content: False

  # Maximum number of tokens allowed in the response.
  max_tokens: 4000
```


## Contributing

Contributions are welcome! Whether it’s adding new features, fixing bugs, or 
improving documentation, here’s how you can help:
- **Report Issues:** Open an issue on GitHub to discuss bugs or ideas.
- **Submit Pull Requests:** Fork the repo, make your changes, and submit a PR.

Please ensure your code follows the existing style and includes appropriate comments.

## Acknowledgments

This project was inspired by the [**"Building a Deep Research AI Agent" course**](https://courses.analyticsvidhya.com/courses/building-a-deep-research-ai-agent) from Analytics Vidhya. 
A huge thank you to the instructors and community for their invaluable lessons and insights into 
building AI-powered research tools.