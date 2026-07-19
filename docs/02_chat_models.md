In LangChain, a Chat Model is a specific type of language model designed for conversations rather than just completing a single sentence.

* `Input/Output Style`: Unlike standard LLMs that take "plain text," Chat Models take a list of messages as input and return a message as output.
* `Message Roles`: It uses three main types of roles to understand context:
     * System Message: Tells the AI how to behave (e.g., "You are a helpful sous-chef").
     * Human Message: What you (the user) type.
     * AI Message: What the model says back.
* `Context Memory`: Because it accepts a list of messages, it is built to remember the "history" of the conversation so it doesn't forget what you said two turns ago.
* `Under the Hood`: While most modern models (like GPT-4 or Cohere’s Command) are technically text-based, LangChain's Chat Model interface wraps them so they are easier to use for building chatbots.

Refer to the section to see available chat models for different LLMs - https://docs.langchain.com/oss/python/integrations/chat
<br>Here we will use `ChatCohere` as we want to interact with cohere model.
<br>click on the model to get more details - https://docs.langchain.com/oss/python/integrations/chat/cohere


## 1. Basic Conversation

### I. The "Message" Structure
LangChain uses specific "message roles" to help the AI understand its instructions versus the user's input
* Here the `messages` variable is a list that acts as the transcript of your conversation. 
* `SystemMessage`: This sets the "persona" or "behavior" of the AI. It’s the behind-the-scenes instruction that the user doesn't usually see, but it dictates how the AI should act (e.g You are a manager assistant to approve or reject leaves..)
     * This is the Instruction Layer.
     * It tells the AI its job (Manager Assistant) and defines the "business logic" (the 3-day rule).
     * The AI uses this to set its "temperature" and decision-making boundaries before looking at your question.
* `HumanMessage`: This represents the actual prompt or question coming from you, the user.
     * This is the Data Layer.
     * It contains the specific user request ("1st Jan to 5th Jan").
     * The AI treats this as the variable it must test against the rules defined in the System Message.
* `AIMessage (Implicit)`: Although not in your code yet, the result you get back is an AIMessage, which represents the AI's response

### II. How model.invoke(messages) Works
When you call .invoke(), you aren't just running a local function; you are triggering a complex Request/Response cycle:
* Serialization: LangChain takes your list of message objects and converts them into a formatted JSON "payload" that the Cohere API understands.
* The API Call: The code sends an HTTPS request over the internet to Cohere's servers. This request includes your API Key (loaded via load_dotenv) and your list of messages.
* Inference: Cohere’s "command-r-plus" model processes the text, predicts the most likely next tokens (words), and generates the answer.
* The Response: The server sends back a data packet. LangChain intercepts this, wraps it in a BaseMessage object, and hands it back to your result variable.
* When you call this, the entire list is bundled together and sent as one package.
* The AI processes the System instructions first to know how to think, then applies that thinking to the Human message to produce the result.

```python
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")


messages = [
    SystemMessage(content="You are a manager assistant to approve or reject leaves. You will be given a leave request and you need to approve or reject it based on the following rules: 1. If the leave request is for more than 3 days, reject it. 2. If the leave request is for less than or equal to 3 days, approve it."),
    HumanMessage(content="can I take leave from 1st Jan to 5th Jan?")
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")
```

III. AI Message
* You don't usually write an AIMessage manually if you are just asking a single question. You use it when you want to provide examples (Few-Shot Prompting) or continue a conversation.
* The best way to use AIMessage in your specific "Manager Assistant" code is to show the AI exactly how you want it to behave by providing a past example before asking the real question.

```python

messages = [
    # 1. THE RULES
   SystemMessage(content="You are a manager assistant to approve or reject leaves. You will be given a leave request and you need to approve or reject it based on the following rules: 1. If the leave request is for more than 3 days, reject it. 2. If the leave request is for less than or equal to 3 days, approve it."),

    # 2. THE EXAMPLE (Human + AI)
    HumanMessage(content="I want 2 days off for a wedding."),
    AIMessage(content="Approved. This request is for 2 days, which is within the 3-day limit."),

    # 3. THE ACTUAL REQUEST
    HumanMessage(content="Can I take leave from 1st Jan to 5th Jan?")
]
```
* Contextual Memory: By adding that AIMessage, you are telling the AI: "Look at how you answered last time. Keep that same tone and logic for the next question."

* Formatting Guide: It shows the AI the exact format you want (e.g., stating the reason before the verdict).

* Consistency: It prevents the AI from "hallucinating" or being too wordy, as it will try to match the style of the previous AIMessage.


## 2. Real time conversation (With history)
In real time when we do a conversation, we remember the context and based on that respond. 
- By default LLMs are stateless means they don't "remember" past prompts. 
- A `chat history` array mimics human memory by storing the transcript.
- With every new query, the entire array is re-sent to the LLM. This provides the necessary context for the model to understand the current intent. This is called contextual injection.
- This allows the LLM to resolve "pronouns" (e.g., knowing "his" refers to the President mentioned in the previous turn). This is called reference resolution.
- So if our first question is who is President of USA? and then next question is tell me his age?
- It will respond to the second question properly as LLM has the context from the first question.
- **Message Roles**: The array isn't just text; it’s categorized into roles (usually SystemMessage, HumanMessage, and AIMessage). This tells the LLM who said what.
- **Token Limit**: As the array grows, it consumes more tokens. If it gets too long, it will hit the LLM's "context window" limit and cause an error or forget the beginning of the chat.
- **Pruning & Summarization**: In advanced setups, we don't pass the entire history forever. We use strategies like Windowing (keeping only the last $k$ messages) or Summarization to keep the history concise.
```console
//Basic flow
User asks Question A -> Store in Array -> LLM Answers A -> User asks Question B -> Send [A + Answer A + B] -> LLM Answers B.
```

```python
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")

chat_history = []

system_message = SystemMessage(content="You are a helpful assistant. Please explain the answer in 100 words or less.")
chat_history.append(system_message)

while True:
    query = input("Enter your query (or 'exit' to quit): ")
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
```

## 3. Saved Chat History
Expand on the below
- Save the chat history to firebase and try
- Memory management technique e.g ConversationBufferMemory keeps everything, while ConversationSummaryMemory summarizes old parts of the chat to save space.
- Windowing (keeping only the last "k" messages) or Summarization