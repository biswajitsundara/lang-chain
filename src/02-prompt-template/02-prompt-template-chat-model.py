from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")


# Approach 1: Prompt with single place holder
template = "tell me a joke about {topic}"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)

final_prompt = prompt.format(topic="cats")
# result = model.invoke(final_prompt)
# print(result.content)


# Approach 2: Prompt with multiple place holders
template = "tell me about {topic} in {language}"
prompt = PromptTemplate(
    input_variables=["topic", "language"],
    template=template,
)
final_prompt = prompt.format(topic="dog", language="Spanish")
# result = model.invoke(final_prompt)
# print(result.content)


# Approach 3: Prompt with placeholders in System and Human messages
# use tuples (role, content)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes")
]

prompt = ChatPromptTemplate.from_messages(messages)
final_prompt = prompt.format_messages(topic="cats", joke_count=3)
result = model.invoke(final_prompt)
print(result.content)