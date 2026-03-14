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
file_path = os.path.join(current_dir, 'data', 'travel_itinerary.txt')
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')

# Check if the database already exists
if not os.path.exists(persistent_directory):
    print(f"Persistent directory doesn't exist. Initializing: {persistent_directory}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    # 1. Load the document
    loader = TextLoader(file_path)
    documents = loader.load()

    # 2. Split the document
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print('-------- Document Chunk Information -----')
    print(f'Number of chunks: {len(docs)}')
    
    # 3. Initialize Cohere Embeddings
    # Since COHERE_API_KEY is in .env, it finds it automatically
    embeddings = CohereEmbeddings(model='embed-v4.0')

    print('------ Creating Vector Store ----')
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )
    print("Success: Vector store created.")

else:
    print(f"Persistent directory already exists: {persistent_directory}")