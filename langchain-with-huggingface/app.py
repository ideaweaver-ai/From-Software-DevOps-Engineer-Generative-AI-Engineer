from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load model locally
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    },
)

# Create prompt template
prompt = PromptTemplate.from_template(
    "You are a helpful assistant.\n\nUser: {input}\n\nAssistant:"
)

# Build chain
chain = prompt | llm | StrOutputParser()

# Invoke
response = chain.invoke({"input": "Explain large language models in simple terms"})
print(response)
