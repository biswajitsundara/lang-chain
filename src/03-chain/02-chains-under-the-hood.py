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