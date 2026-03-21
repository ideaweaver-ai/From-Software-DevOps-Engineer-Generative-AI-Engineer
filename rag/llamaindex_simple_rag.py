# pip install pypdf openai llama-index llama-index-embeddings-openai llama-index-llms-openai

import os
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Set your OpenAI API key in environment before running
# export OPENAI_API_KEY="your-openai-key"

pdf_path = "ideaweaver_policy_doc.pdf"
persist_dir = "./storage_openai"

# LlamaIndex settings
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=150)
Settings.llm = OpenAI(model="gpt-4.1-mini", temperature=0)

# Load the PDF
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

# Build index only once, otherwise load from disk
if not os.path.exists(persist_dir):
    print("No saved index found. Creating a new index...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    print("Saved index found. Loading index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=4)

# Ask a question
question = input("Ask a question: ")

# Get answer
response = query_engine.query(question)

print("\nAnswer:")
print(response.response)
