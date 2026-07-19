A Chain is a sequence of steps where the output of one step becomes the input of the next step.

* It connects multiple components like prompts, LLMs, tools, or retrievers.
* Used to build workflows for AI applications.
* Helps organize complex tasks into smaller steps.
* Example flow:
     * User Input → Prompt Template → LLM → Output
* Improves reusability and modularity in AI pipelines.
* Chain = Step-by-step pipeline for LLM tasks.

## 1. Basic Example
* **Without a chain**: You must manually format the prompt, call the API, parse the JSON response, and extract the result.
* **With a chain**: All these steps are wrapped into one reusable workflow, making the code simpler and cleaner.
* **StrOutputParser**: By default, LLMs return complex objects. This "link" in the chain extracts just the text string for you.
* **The | Operator**: Enables chaining by passing the output of one step directly as the input to the next step.

```python
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")

prompt = ChatPromptTemplate.from_template("Tell me a short, witty joke about {topic}.")


# Define the Chain using the pipe (|) operator
# This flow is: Input -> Prompt -> Model -> String Output

chain = prompt | model | StrOutputParser()


# Run the Chain
response = chain.invoke({"topic": "programming"})

print(response)
```

## 2. Chains under the hood
In LangChain, almost every component (Prompts, Models, Parsers) inherits from a class called `Runnable`.
* When you use the pipe operator (|), you are actually creating a RunnableSequence.
* Think of a `Runnable` as a unit of work with a specific input and output type. 
* A `RunnableSequence` simply takes the output of `Runnable A` and feeds it as the input to `Runnable B`.

```python
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence


load_dotenv()


# 1. The Components
model = ChatCohere(model="command-r-plus-08-2024")
prompt = ChatPromptTemplate.from_template("Tell me a short, witty joke about {topic}.")
parser = StrOutputParser()


# 2. The runnables
format_step = RunnableLambda(lambda x: prompt.invoke(x))
model_step = RunnableLambda(lambda x: model.invoke(x))
parse_step = RunnableLambda(lambda x: parser.invoke(x))


# 3. RunnableSequence
chain = RunnableSequence(format_step, model_step, parse_step)


# 4. Run the Chain
# The dictionary {"topic": "programming"} enters the first Runnable.
# Its output (a PromptValue) enters the second, and so on.
response = chain.invoke({"topic": "programming"})

print(response)
```

## 3. LangChain Expression Language (LCEL)
LCEL lets you describe the flow of data between components in a simple expression style.
* LCEL is a syntax for composing and connecting runnables to build chains.
* It allows you to combine prompts, LLMs, tools, and parsers in a pipeline.
* Uses operators like | to pass output of one step to the next.
* Makes chain creation declarative, readable, and modular.
* Built on top of the Runnable interface.
* LCEL = a concise way to build LangChain pipelines by composing runnables.

```python
chain = prompt | llm | parser

# Input → PromptTemplate → LLM → OutputParser → Result
```