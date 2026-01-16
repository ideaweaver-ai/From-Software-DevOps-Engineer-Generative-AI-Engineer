import streamlit as st
from transformers import pipeline

st.title("Text Generator")

# Using Qwen2.5-0.5B-Instruct - efficient 500M parameter model
# Good balance between size and quality
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )

pipe = load_model()

text = st.text_input("Enter text to complete or ask a question")

if st.button("Generate"):
    if text:
        with st.spinner("Generating..."):
            # Qwen2.5-Instruct works well with direct text or formatted prompts
            # For questions, format as instruction; for completion, use directly
            if text.strip().endswith("?"):
                prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            else:
                # For text completion, just use the text directly
                prompt = text
            
            output = pipe(
                prompt,
                max_length=len(prompt.split()) + 100,  # Generate more text
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=pipe.tokenizer.eos_token_id
            )
            result = output[0]["generated_text"]
            
            # Clean up the result - remove the prompt if it was included
            if prompt in result:
                result = result.replace(prompt, "").strip()
            
            st.write("**Generated Text:**")
            st.write(result)
