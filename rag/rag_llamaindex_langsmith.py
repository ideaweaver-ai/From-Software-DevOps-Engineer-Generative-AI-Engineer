# pip install pypdf openai langsmith llama-index llama-index-embeddings-openai llama-index-llms-openai

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

from langsmith import traceable

# --------------------------------------------------
# Environment variables you need
# --------------------------------------------------
# export OPENAI_API_KEY="your-openai-key"
# export LANGSMITH_API_KEY="your-langsmith-key"
# export LANGSMITH_TRACING="true"
# export LANGSMITH_PROJECT="llamaindex-rag-demo"

# Optional, if your LangSmith API key belongs to multiple workspaces:
# export LANGSMITH_WORKSPACE_ID="your-workspace-id"

pdf_path = "./google_adwords_for_donations.pdf"
persist_dir = "./storage_openai"

# --------------------------------------------------
# LlamaIndex settings
# --------------------------------------------------
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

Settings.node_parser = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=150
)

Settings.llm = OpenAI(
    model="gpt-4.1-mini",
    temperature=0.2
)

# --------------------------------------------------
# Traced functions
# --------------------------------------------------
@traceable(name="load_pdf", run_type="tool")
def load_documents(file_path: str):
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    print(f"Loaded {len(docs)} document(s).")
    return docs


@traceable(name="build_index", run_type="chain")
def build_and_save_index(docs, save_dir: str):
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=save_dir)
    print(f"Index created and saved to {save_dir}")
    return index


@traceable(name="load_index", run_type="chain")
def load_saved_index(save_dir: str):
    storage_context = StorageContext.from_defaults(persist_dir=save_dir)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully.")
    return index


@traceable(name="ask_question", run_type="chain")
def ask_index(index, question: str):
    query_engine = index.as_query_engine(similarity_top_k=4)
    response = query_engine.query(question)
    return response.response


@traceable(name="rag_pipeline", run_type="chain")
def run_pipeline():
    documents = load_documents(pdf_path)

    if not os.path.exists(persist_dir):
        print("No saved index found. Creating a new index...")
        index = build_and_save_index(documents, persist_dir)
    else:
        print("Saved index found. Loading index from disk...")
        index = load_saved_index(persist_dir)

    question = "What is the primary purpose of Google AdWords for non-profits as discussed in the document?"
    answer = ask_index(index, question)

    print("\nQuestion:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    run_pipeline()
