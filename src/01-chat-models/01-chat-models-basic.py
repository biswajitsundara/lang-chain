from dotenv import load_dotenv
from langchain_cohere import ChatCohere

load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")

result = model.invoke("What is the capital of France?")

#print(result)
print(result.content)