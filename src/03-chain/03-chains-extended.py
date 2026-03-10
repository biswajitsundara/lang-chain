from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")

prompt = ChatPromptTemplate.from_template("Tell me a short, witty joke about {topic}.")


upper_case_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}")
#combined_output = RunnableLambda(lambda x: f"JOKE:\={x}\n\nWord count: {len(x.split())}")

chain = prompt | model | StrOutputParser() | upper_case_output | count_words


# Run the Chain
response = chain.invoke({"topic": "programming"})

print(response)