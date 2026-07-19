"""
Understanding the chains in LangChain v1
LCEL patterns, composition and debugging
"""

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()
llm = init_chat_model("cohere:command-r-plus-08-2024", temperature=0.5, max_tokens=500)

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    parser = StrOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"text": "LangChain is a framework for developing applications powered by language models. It provides a standard interface for chains, allowing developers to combine different components and create complex workflows."})
    print(f"Summary: {result}")


def demo_parallel_chain():

    summarize_prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    keywords_prompt = ChatPromptTemplate.from_template("Extract 5 keywords from the following text: {text}\n Return the keywords as a comma-separated list.")
    sentiment_prompt = ChatPromptTemplate.from_template("What is the sentiment of the following text: {text}")
    parser = StrOutputParser()
    
    #Parallel execution of chains
    parallel_chain = RunnableParallel(
        summarize=summarize_prompt | llm | parser,
        keywords=keywords_prompt | llm | parser,
        sentiment=sentiment_prompt | llm | parser
    )
    
    result = parallel_chain.invoke({"text": "LangChain is a framework for developing applications powered by language models. It provides a standard interface for chains, allowing developers to combine different components and create complex workflows."})
    print(f"Summary: {result['summarize']}") 
    print("-"*50) 
    print(f"Keywords: {result['keywords']}")
    print("-"*50) 
    print(f"Sentiment: {result['sentiment']}")

if __name__ == "__main__":
    #demo_basic_chain()
    demo_parallel_chain()