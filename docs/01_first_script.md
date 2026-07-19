### Code Walkthrough
* `load_dotenv()` — loads your COHERE_API_KEY from a .env file into environment variables
* `ChatCohere` — LangChain wrapper to interact with Cohere's chat models
* `llm = ChatCohere(...)` — initializes the `command-r-plus-08-2024` model with temperature=0 (deterministic output)
* `llm.invoke(...)` — sends a prompt to the model and gets a response
* `if __name__ == "__main__" — ensures main()` only runs when the script is executed directly

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_core import __version__ as core_version
from langchain_cohere import ChatCohere

print(f"langchain-core version: {core_version}")


def main():
    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
    response = llm.invoke("What is the capital of France? in one word")
    print(f"Response from cohere: {response}")


if __name__ == "__main__":
    main()

```