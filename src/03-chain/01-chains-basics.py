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