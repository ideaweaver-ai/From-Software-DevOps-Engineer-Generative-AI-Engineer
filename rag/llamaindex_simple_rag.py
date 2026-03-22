# pip install pypdf openai faiss-cpu llama-index llama-index-embeddings-openai llama-index-llms-openai llama-index-vector-stores-faiss

import os
import faiss

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore

pdf_path = "ideaweaver_policy_doc.pdf"
persist_dir = "./storage_faiss"

# LlamaIndex settings
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=150)
Settings.llm = OpenAI(model="gpt-4.1-mini", temperature=0)

# Load documents
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

# text-embedding-3-small produces 1536-dimensional embeddings
dimension = 1536

if not os.path.exists(persist_dir):
    print("No saved FAISS index found. Creating a new index...")

    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(dimension)

    # Wrap with LlamaIndex FAISS vector store
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # Persist everything
    index.storage_context.persist(persist_dir=persist_dir)

else:
    print("Saved FAISS index found. Loading index from disk...")

    # Reload FAISS vector store from disk
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)

    # Rebuild storage context using the persisted FAISS store
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=persist_dir,
    )

    # Load index
    index = load_index_from_storage(storage_context)

# Query engine
query_engine = index.as_query_engine(similarity_top_k=4)

question = input("Ask a question: ")
response = query_engine.query(question)

print("\nAnswer:")
print(response.response)
