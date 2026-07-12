# ============================================================
# Apache Log Analyzer - Built with LangChain + OpenAI
# ============================================================
# This app reads an Apache log file, extracts error entries,
# and sends them to GPT for analysis.
#
# Workflow: parse logs -> analyze errors -> print result
# ============================================================

import os
import re
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

BATCH_SIZE = 50


# ------------------------------------------------------------
# STEP 1: Parse the log file (plain Python — no LangChain needed)
# ------------------------------------------------------------

def parse_logs(log_file_path: str) -> list[str]:
    """Read the log file and return lines that look like errors."""
    error_lines = []

    with open(log_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # HTTP status >= 400 (e.g. 404, 500)
            status_match = re.search(r'" (\d{3}) ', line)
            if status_match and int(status_match.group(1)) >= 400:
                error_lines.append(line)
                continue

            # Keywords like "fatal", "segfault"
            lower_line = line.lower()
            if any(word in lower_line for word in ["error", "fatal", "segfault", "exit signal"]):
                error_lines.append(line)

    print("\n--- Log Parser ---")
    print(f"Read file: {log_file_path}")
    print(f"Total error lines found: {len(error_lines)}")

    return error_lines


# ------------------------------------------------------------
# STEP 2: Set up the LLM and prompt templates (LangChain part)
# ------------------------------------------------------------
# ChatPromptTemplate lets you define prompts with placeholders.
# {errors_text} gets filled in when you call .invoke(...)
# ------------------------------------------------------------

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

analysis_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """You are a senior DevOps engineer analyzing Apache web server logs.

Below are error log entries (batch {batch_num} of {total_batches}).
Please provide:

1. **Error Summary**: Table grouping errors by type with counts
2. **Root Cause Analysis**: Likely cause for each category
3. **Severity Assessment**: Critical, High, Medium, or Low
4. **Recommended Fixes**: Specific actionable steps
5. **Priority Order**: What to fix first and why

Error log entries:

{errors_text}

Provide a clear, well-organized analysis.""",
    )
])

combine_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """You are a senior DevOps engineer. Below are separate analyses
of Apache error logs processed in {total_batches} batches.
Combine them into ONE unified report with:

1. **Error Summary**: Combined table with total counts
2. **Root Cause Analysis**: Merged per category
3. **Severity Assessment**: Overall severity
4. **Recommended Fixes**: Consolidated steps
5. **Priority Order**: Final priority ranking

Batch analyses:

{combined_text}

Provide a clear COMBINED analysis.""",
    )
])

# A "chain" = prompt template -> LLM (the | operator pipes output forward)
analyze_chain = analysis_prompt | llm
combine_chain = combine_prompt | llm


# ------------------------------------------------------------
# STEP 3: Analyze errors with batching
# ------------------------------------------------------------

def analyze_errors(error_lines: list[str]) -> str:
    """Send error lines to GPT and return the final analysis."""
    if not error_lines:
        return "No errors found in the log file!"

    batches = [
        error_lines[i : i + BATCH_SIZE]
        for i in range(0, len(error_lines), BATCH_SIZE)
    ]
    total_batches = len(batches)

    print("\n--- LLM Analysis ---")
    print(f"Total errors: {len(error_lines)}")
    print(f"Number of batches: {total_batches}")

    batch_results = []
    for batch_num, batch in enumerate(batches, start=1):
        print(f"Processing batch {batch_num}/{total_batches}...")

        errors_text = "\n".join(batch)

        response = analyze_chain.invoke({
            "batch_num": batch_num,
            "total_batches": total_batches,
            "errors_text": errors_text,
        })
        batch_results.append(response.content)

    if total_batches == 1:
        return batch_results[0]

    print(f"\nCombining {total_batches} batch analyses...")
    combined_text = ""
    for i, result in enumerate(batch_results, start=1):
        combined_text += f"\n--- Batch {i} Analysis ---\n{result}\n"

    final_response = combine_chain.invoke({
        "total_batches": total_batches,
        "combined_text": combined_text,
    })
    return final_response.content


# ------------------------------------------------------------
# STEP 4: Main — just call the functions in order
# ------------------------------------------------------------

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Please set your OpenAI API key first!")
        print()
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        print()
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "apache.log")

    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found at {log_file}")
        sys.exit(1)

    print("=" * 60)
    print("  Apache Log Analyzer")
    print("  Powered by LangChain + OpenAI GPT")
    print("=" * 60)

    error_lines = parse_logs(log_file)
    analysis = analyze_errors(error_lines)

    print("\n" + "=" * 60)
    print("  ANALYSIS RESULT")
    print("=" * 60 + "\n")
    print(analysis)
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
