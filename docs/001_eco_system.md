## The Langchain Ecosystem

### Core Packages
1. `langchain-core` : Core abstraction and LCEL
2. `langchain`: Agents, chains, high-level APIs
3. `langgraph`: Stateful agent orchestration
4. `langsmith`: Tracing, evaluation, monitoring
5. `langserve`: Deploy as REST APIs

### Integration Packages
1. `langchain-openai`: Open AI integration
2. `langchain_cohere`: Cohere AI integration
3. `langchain-anthropic`: Anthropic/claude integration
4. `langchain-community`: Community integrations

### When to use what
1. `LangChain` : Building chains, RAG, quick prototypes
2. `LangGraph` : Stateful agents, loops, multi-agent, production
3. `LangSmith`: Debugging, monitoring, evaluation
4. `LangServe`: Deploying as API

### Benefits of using Langchain/ this eco system
* We can swap/migrate to other AI models easily
* If we have developped a solution using open AI
* Then we can just change the model to Anthropic without making any other changes
