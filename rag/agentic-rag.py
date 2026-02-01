# pip install pypdf sentence-transformers transformers llama-index \
#   llama-index-embeddings-huggingface llama-index-llms-huggingface

"""
Agentic RAG Implementation using LlamaIndex

This demonstrates how to convert traditional RAG into Agentic RAG where:
1. An agent makes autonomous decisions about which tools to use
2. Multiple specialized query engines act as tools
3. The router intelligently selects the appropriate engine based on query type
"""

from llama_index.core import VectorStoreIndex, SummaryIndex, Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# ============================================================================
# STEP 1: Load and Prepare Documents (Same as vanilla RAG)
# ============================================================================

pdf_path = "./google_adwords_for_donations.pdf"

# Load PDF documents
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
print(f"✓ Loaded {len(documents)} document(s).")

# ============================================================================
# STEP 2: Configure Embeddings (Same as vanilla RAG)
# ============================================================================

# Use HuggingFace embeddings for semantic search
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("✓ Embedding model configured.")

# ============================================================================
# STEP 3: Configure Chunking Strategy (Same as vanilla RAG)
# ============================================================================

# Reduce chunk size for TinyLlama's small context window
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
print("✓ Chunking strategy configured.")

# ============================================================================
# STEP 4: Configure LLM (Same as vanilla RAG)
# ============================================================================

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
    max_new_tokens=128,  # Reduced from 256
    generate_kwargs={"temperature": 0.2, "do_sample": False},
    device_map="auto",
)
print("✓ LLM configured.")

# ============================================================================
# STEP 5: Create Multiple Specialized Indices (KEY DIFFERENCE!)
# ============================================================================
# In vanilla RAG, we only create one VectorStoreIndex.
# In Agentic RAG, we create MULTIPLE indices for different purposes.

# Index 1: Vector Index for detailed, specific queries
vector_index = VectorStoreIndex.from_documents(documents)
print("✓ Vector index created (for detailed queries).")

# Index 2: Summary Index for high-level overview queries
summary_index = SummaryIndex.from_documents(documents)
print("✓ Summary index created (for overview queries).")

# ============================================================================
# STEP 6: Create Specialized Query Engines (KEY DIFFERENCE!)
# ============================================================================
# Each index becomes a query engine - these are the TOOLS the agent can use.

# Vector Query Engine: For specific, detailed questions
# Reduce similarity_top_k to retrieve fewer chunks
vector_query_engine = vector_index.as_query_engine(
    similarity_top_k=2,  # Reduced from 4
    response_mode="compact"
)

# Summary Query Engine: For overview and summarization questions
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"
)

print("✓ Query engines created.")

# ============================================================================
# STEP 7: Wrap Query Engines as Tools (KEY DIFFERENCE!)
# ============================================================================
# This is where we define what each tool does. The agent reads these
# descriptions to decide which tool to use for a given query.

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="vector_search",
    description=(
        "Useful for answering SPECIFIC questions that require detailed information. "
        "Use this when the user asks 'how to', 'what is the exact', 'show me steps', "
        "or needs precise details about implementation, configuration, or technical specifics. "
        "Examples: 'How do I set up Google AdWords?', 'What are the specific requirements?'"
    )
)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="summary_search",
    description=(
        "Useful for answering HIGH-LEVEL questions that need overviews or summaries. "
        "Use this when the user asks for 'overview', 'summarize', 'what are the main', "
        "or wants a general understanding without deep technical details. "
        "Examples: 'What is this document about?', 'Summarize the key benefits', 'Give me an overview'"
    )
)

print("✓ Tools created with descriptions.")

# ============================================================================
# STEP 8: Create the Router (THE AGENT!) (KEY DIFFERENCE!)
# ============================================================================
# The RouterQueryEngine is the intelligent agent that makes autonomous decisions.
# It reads the query, analyzes it, and decides which tool to use.

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),  # Uses LLM to make routing decisions
    query_engine_tools=[vector_tool, summary_tool],  # Available tools
    verbose=True  # Show which tool is being selected
)

print("✓ Router Query Engine (Agent) created.")
print("=" * 80)
print("AGENTIC RAG SYSTEM READY")
print("=" * 80)

# ============================================================================
# STEP 9: Test with Different Query Types
# ============================================================================
# The agent will automatically route to the appropriate engine based on query type.

print("\n" + "=" * 80)
print("TEST 1: DETAILED QUERY (Should route to vector_search)")
print("=" * 80)

detailed_question = (
    "What is the primary purpose of Google AdWords for non-profits "
    "as discussed in the document?"
)
print(f"\nQuestion: {detailed_question}")
print("\nAgent is analyzing query and selecting tool...")
response1 = router_query_engine.query(detailed_question)
print(f"\nAnswer: {response1.response}")
