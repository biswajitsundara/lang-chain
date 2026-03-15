import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader 
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

# Load environment variables from .env
load_dotenv()

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, 'data', 'books')
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, 'chroma_db_metadata')

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print(f"Persistent directory doesn't exist. Initializing: {persistent_directory}")
    
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Books directory not found at: {books_dir}")
    
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]
    
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {'source': book_file}  # Add the filename as source metadata
            documents.append(doc)
        
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    relevant_docs = text_splitter.split_documents(documents)

    print('-------- Document Chunk Information -----')
    print(f'Number of chunks: {len(relevant_docs)}')

    for i, doc in enumerate(relevant_docs, 1):
        print(f"document{i} : {doc.page_content}\n\n")
        print(f"source : {doc.metadata['source']}\n\n")

    # Fixed: Embeddings and Vectorstore must be inside the 'if' block to use relevant_docs
    embeddings = CohereEmbeddings(model='embed-v4.0')

    print('------ Creating Vector Store ----')
    vectorstore = Chroma.from_documents(
        documents=relevant_docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )
    print("Success: Vector store created.")

else:
    print(f"Persistent directory already exists: {persistent_directory}")
    # Initialize the existing store if needed
    embeddings = CohereEmbeddings(model='embed-v4.0')
    vectorstore = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)