from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  

def basic_chain():
    """
    A basic chain that takes a question as input and returns an answer in one sentence.
    """
    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant, answer in one sentence {question}")
    parser = StrOutputParser()

    chain = prompt | llm | parser
    result = chain.invoke({"question": "What is lang chain?"})
    print(result)
    return chain

if __name__ == "__main__":
    basic_chain()