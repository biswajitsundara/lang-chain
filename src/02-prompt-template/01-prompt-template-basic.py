from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")


# Approach 1: Using string formatting
template = "tell me a joke about {topic}"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)

final_prompt = prompt.format(topic="cats")
print("-- Approach 1 ---\n")
print(final_prompt)


# Approach 2: Using from_template method
prompt = PromptTemplate.from_template("What is the capital of {country}?")
print("-- Approach 2 ---\n")
print(prompt.format(country="France"))



# Approach 3: Using from_template method
template = "tell me about {topic} in {language}"
prompt = PromptTemplate(
    input_variables=["topic", "language"],
    template=template,
)
final_prompt = prompt.format(topic="dog", language="Spanish")
print("-- Approach 3 ---\n")
print(final_prompt)



# Approach 4: Using from_messages method
# use tuples (role, content)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes")
]

prompt = ChatPromptTemplate.from_messages(messages)
final_prompt = prompt.format_messages(topic="cats", joke_count=3)
print("-- Approach 4 ---\n")
print(final_prompt)



# Approach 5: Using from_messages method
# use tuples (role, content)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    HumanMessage(content="Tell me 3 jokes")
]

prompt = ChatPromptTemplate.from_messages(messages)
final_prompt = prompt.format_messages(topic="cats")
print("-- Approach 5 ---\n")
print(final_prompt)


# Approach 6: This does not work
# The final prompt will have "Tell me {joke_count} jokes"
messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    HumanMessage(content="Tell me {joke_count} jokes")
]
prompt = ChatPromptTemplate.from_messages(messages)
final_prompt = prompt.format_messages(topic="cats", joke_count=3)
print("-- Approach 6 ---\n")
print(final_prompt)