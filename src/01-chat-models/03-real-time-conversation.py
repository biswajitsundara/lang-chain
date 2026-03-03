from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")

chat_history = []

system_message = SystemMessage(content="You are a helpful assistant. Please explain the answer in 100 words or less.")
chat_history.append(system_message)

while True:
    query = input("Enter your query (or 'exit'): ")
    if query.lower() == 'exit':
        break   
    human_message = HumanMessage(content=query)
    chat_history.append(human_message)

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

print("Conversation ended.")
print(chat_history)