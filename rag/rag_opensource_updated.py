# pip install pypdf sentence-transformers transformers llama-index \
#   llama-index-embeddings-huggingface llama-index-llms-huggingface

import os
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# -----------------------------
# File paths
# -----------------------------
pdf_path = "./google_adwords_for_donations.pdf"
persist_dir = "./storage"

# -----------------------------
# Load PDF
# -----------------------------
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
print(f"Loaded {len(documents)} document(s).")

# -----------------------------
# Embedding model
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# -----------------------------
# Chunking settings
# -----------------------------
Settings.node_parser = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=150
)

# -----------------------------
# LLM settings
# -----------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

query_wrapper_prompt = PromptTemplate(
    "<|system|>\nYou are a helpful AI assistant.\n"
    "<|user|>\n{query_str}\n"
    "<|assistant|>\n"
)

Settings.llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer_name=model_name,
    query_wrapper_prompt=query_wrapper_prompt,
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={
        "temperature": 0.2,
        "do_sample": False
    },
    device_map="auto",
)

# -----------------------------
# Create or load index
# -----------------------------
if not os.path.exists(persist_dir):
    print("No saved index found. Creating a new index...")

    index = VectorStoreIndex.from_documents(documents)

    # Save index locally
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index created and saved to {persist_dir}")

else:
    print("Saved index found. Loading index from disk...")

    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully.")

# -----------------------------
# Query
# -----------------------------
query_engine = index.as_query_engine(similarity_top_k=4)

question = "What is the primary purpose of Google AdWords for non-profits as discussed in the document?"
response = query_engine.query(question)

print("\nQuestion:", question)
print("Answer:", response.response)
