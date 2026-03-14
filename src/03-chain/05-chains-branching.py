from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()
model = ChatCohere(model="command-r-plus-08-2024")

# 1. Prompts (Keeping your original templates)
positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides positive feedback."),
    ("human", "Generate a thank you note for the feedback: {feedback}.")
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that addresses negative feedback."),
    ("human", "Generate a response addressing this negative feedback: {feedback}.")
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that handles neutral feedback."),
    ("human", "Generate a request for more details for this neutral feedback: {feedback}.")
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that escalates feedback."),
    ("human", "Generate a message to escalate this feedback to an human agent: {feedback}.")
])

classification_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that classifies customer feedback into categories: positive, negative, neutral, or escalate."),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.")   
])

# 2. Branches
# Note: x now represents a dictionary containing "classification" and "feedback"
branches = RunnableBranch(
    (
        lambda x: "positive" in x["classification"].lower(),
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x["classification"].lower(),
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x["classification"].lower(),
        neutral_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "escalate" in x["classification"].lower(),
        escalate_feedback_template | model | StrOutputParser()
    ),
    # Default fallback
    neutral_feedback_template | model | StrOutputParser()
)

# 3. Classification Chain
classification_chain = (
    classification_feedback_template 
    | model 
    | StrOutputParser() 
)

# 4. Final Chain
chain = (
    RunnableParallel(
        classification=classification_chain,
        feedback=lambda x: x["feedback"] 
    )
    | branches
)

#review = "I love the new MacBook Pro! The performance is amazing and the battery life is fantastic."
review = "The mac book pro is slow, crashes all the time, and the battery dies after just a few hours."

# 5. Invoke
result = chain.invoke({"feedback": review})
print(result)