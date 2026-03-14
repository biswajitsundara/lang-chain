from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

"""
Here the two branches (pros and cons) run in parallel after the initial model call that extracts the features. 
The outputs from both branches are then combined into a single string format for the final output.
"""
load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")

# 1. Initial Prompt - Using the (role, content) tuple format
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that analyzes products."),
    ("human", "List down three main features of the {product}.")
])

# 2. Refined Branch Prompts
pros_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert product reviewer."),
    ("human", "Given these features: {features}, list the pros of this product.")
])

cons_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert product reviewer."),
    ("human", "Given these features: {features}, list the cons of this product.")
])

def combine_pros_cons(input_dict):
    return f"### PROS:\n{input_dict['branches']['pros']}\n\n### CONS:\n{input_dict['branches']['cons']}"

# 3. Define the parallel branches
# We map the string from the first model into a dictionary for the next prompt
pros_branch = (
    RunnableLambda(lambda x: {"features": x}) | pros_prompt | model | StrOutputParser()
)
cons_branch = (
    RunnableLambda(lambda x: {"features": x}) | cons_prompt | model | StrOutputParser()
)

# 4. Assemble the chain
chain = (
    prompt_template 
    | model 
    | StrOutputParser() 
    | RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch}) 
    | RunnableLambda(combine_pros_cons)
)

# 5. Invoke
result = chain.invoke({"product": "MacBook Pro"})
print(result)