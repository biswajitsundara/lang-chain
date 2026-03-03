from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")


# messages = [
#     SystemMessage(content="You are a manager assistant to approve or reject leaves. You will be given a leave request and you need to approve or reject it based on the following rules: 1. If the leave request is for more than 3 days, reject it. 2. If the leave request is for less than or equal to 3 days, approve it."),
#     HumanMessage(content="can I take leave from 1st Jan to 5th Jan?")
# ]

#result = model.invoke(messages)
#print(f"Answer from AI: {result.content}")



messages = [
    # 1. THE RULES
   SystemMessage(content="You are a manager assistant to approve or reject leaves. You will be given a leave request and you need to approve or reject it based on the following rules: 1. If the leave request is for more than 3 days, reject it. 2. If the leave request is for less than or equal to 3 days, approve it."),

    # 2. THE EXAMPLE (Human + AI)
    HumanMessage(content="I want 2 days off for a wedding."),
    AIMessage(content="Approved. This request is for 2 days, which is within the 3-day limit."),

    # 3. THE ACTUAL REQUEST
    HumanMessage(content="Can I take leave from 1st Jan to 5th Jan?")
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")