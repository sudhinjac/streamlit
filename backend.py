import os
import sys
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Enable detailed logging
#logging.basicConfig(level=logging.DEBUG)

# Paths (using WSL-native directories)
local_path = "/mnt/c/testlama/PDF"
vector_db_path = "/home/sudhin/vector_db"

print(f"Local Path: {os.path.abspath(local_path)}")
print(f"Vector DB Path: {os.path.abspath(vector_db_path)}")

# Ensure the vector_db_path exists
try:
    os.makedirs(vector_db_path, exist_ok=True)
    print(f"Directory '{vector_db_path}' is ready.")
except Exception as e:
    print(f"Error creating directory '{vector_db_path}': {e}")
    sys.exit(1)

# Verify write permissions
test_file = os.path.join(vector_db_path, "test_write_permissions.txt")
try:
    with open(test_file, 'w') as f:
        f.write("Testing write permissions.")
    os.remove(test_file)
    print("Write permissions confirmed.")
except Exception as e:
    print(f"Write permissions error in '{vector_db_path}': {e}")
    sys.exit(1)

# Load and process PDF files
chunks = []
for filename in os.listdir(local_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(local_path, filename)
        print(f"Loading file: {file_path}")
        loader = UnstructuredPDFLoader(file_path=file_path)
        try:
            data = loader.load()
            print(f"Loaded {len(data)} documents from {filename}.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Add metadata to identify the source document
        for doc in data:
            doc.metadata["source"] = filename

        # Use a refined chunk size with overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        try:
            split_docs = text_splitter.split_documents(data)
            chunks.extend(split_docs)
            print(f"Split into {len(split_docs)} chunks.")
        except Exception as e:
            print(f"Error splitting documents from {filename}: {e}")

print(f"Total chunks to embed: {len(chunks)}")

# Set up embeddings and vector store
embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
try:
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory=vector_db_path,
        collection_name="local-rag"
    )
    print("Vector store created successfully!")
except Exception as e:
    print(f"Error creating vector store: {e}")
    sys.exit(1)