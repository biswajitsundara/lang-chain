import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

# 1. Setup Environment
load_dotenv()

# 2. Define the missing variables
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db_metadata')

# Use your high-end v4.0 model
embeddings = CohereEmbeddings(model='embed-v4.0')

# 3. Initialize the DB
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# 4. Define a query based on your travel plan
query = "What happened at the Rialto Bridge?"

# 5. Setup the Retriever (Note: search_type fix)
retriever = db.as_retriever(
   search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.01} # Start near zero and move up
)

# 6. Invoke
relevant_docs = retriever.invoke(query)

# 7. Print Results
print(f"--- Results for: {query} ---")
if not relevant_docs:
    print("No documents matched the similarity threshold.")
else:
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\nDocument {i}: {doc.page_content}")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")