# ============================================================
# Apache Log Analyzer - Built with LangGraph + OpenAI
# ============================================================
# This app reads an Apache log file, extracts error entries,
# and sends them to GPT for analysis.
#
# Workflow: START -> parse_logs -> analyze_errors -> END
# ============================================================

import re
from typing import TypedDict


# ------------------------------------------------------------
# STEP 1: Define the State
# ------------------------------------------------------------
# In LangGraph, "state" is the data that flows between nodes.
# Think of it as a shared dictionary that each node can read
# from and write to.
#
# We use TypedDict so Python knows what fields to expect.
# ------------------------------------------------------------

class LogAnalysisState(TypedDict):
    log_file_path: str            # Path to the Apache log file
    error_lines: list[str]        # Parsed error lines from the log
    batch_analyses: list[str]     # Results from each batch (for large files)
    analysis: str                 # Final combined LLM analysis result


# ------------------------------------------------------------
# STEP 2: Create the Log Parser Node
# ------------------------------------------------------------
# A "node" in LangGraph is just a Python function.
# It receives the current state and returns a dictionary
# with the fields it wants to update.
#
# This node reads the log file and picks out error lines.
# An error is any line where:
#   - HTTP status code is 400 or higher (like 404, 500, 503)
#   - OR the line contains keywords: error, fatal, segfault
# ------------------------------------------------------------

def parse_logs(state: LogAnalysisState) -> dict:
    """Read the log file and extract error lines."""

    error_lines = []

    with open(state["log_file_path"], "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check 1: Look for HTTP status codes >= 400
            # In Apache logs, the status code appears after the request
            # Example: "GET /page HTTP/1.1" 404 196
            #                                ^^^ this is the status code
            status_match = re.search(r'" (\d{3}) ', line)
            if status_match and int(status_match.group(1)) >= 400:
                error_lines.append(line)
                continue

            # Check 2: Look for error keywords (PHP errors, segfaults, etc.)
            lower_line = line.lower()
            if any(word in lower_line for word in ["error", "fatal", "segfault", "exit signal"]):
                error_lines.append(line)

    # Print a summary so we can see the node working
    print(f"\n--- Log Parser Node ---")
    print(f"Read file: {state['log_file_path']}")
    print(f"Total error lines found: {len(error_lines)}")

    # Return only the fields we want to update in the state
    return {"error_lines": error_lines}


# ------------------------------------------------------------
# STEP 3: Create the LLM Analysis Node
# ------------------------------------------------------------
# This node takes the error lines from the state and sends
# them to OpenAI's GPT model for analysis.
#
# We use langchain_openai's ChatOpenAI class, which makes it
# easy to call OpenAI models. It automatically reads the
# OPENAI_API_KEY from your environment variables.
# ------------------------------------------------------------

# How many error lines to send to GPT in one batch.
# If you have more errors than this, they'll be split into
# multiple batches and analyzed separately, then combined.
BATCH_SIZE = 50


def analyze_errors(state: LogAnalysisState) -> dict:
    """Send the error lines to GPT for analysis (with batching)."""
    from langchain_openai import ChatOpenAI

    # If no errors were found, no need to call the LLM
    if not state["error_lines"]:
        return {"analysis": "No errors found in the log file!"}

    # Create the LLM instance
    # - model: gpt-4o-mini is fast and cheap, great for learning
    # - temperature: 0 means deterministic (same input = same output)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Split error lines into batches of BATCH_SIZE
    # Example: 120 errors with BATCH_SIZE=50 -> 3 batches (50, 50, 20)
    error_lines = state["error_lines"]
    batches = []
    for i in range(0, len(error_lines), BATCH_SIZE):
        batches.append(error_lines[i : i + BATCH_SIZE])

    total_batches = len(batches)
    print(f"\n--- LLM Analysis Node ---")
    print(f"Total errors: {len(error_lines)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {total_batches}")

    # Process each batch through GPT
    batch_results = []
    for batch_num, batch in enumerate(batches, start=1):
        print(f"\nProcessing batch {batch_num}/{total_batches}...")

        errors_text = "\n".join(batch)

        prompt = f"""You are a senior DevOps engineer analyzing Apache web server logs.

Below are the error log entries extracted from an Apache server (batch {batch_num} of {total_batches}).
Please analyze them and provide:

1. **Error Summary**: A table grouping errors by type (e.g., 404, 500, PHP Fatal, Segfault, etc.) with counts
2. **Root Cause Analysis**: For each error category, explain what is likely causing it
3. **Severity Assessment**: Rate each category as Critical, High, Medium, or Low
4. **Recommended Fixes**: Specific, actionable steps to fix each category of errors
5. **Priority Order**: Which errors should be fixed first and why

Here are the error log entries:

{errors_text}

Provide a clear, well-organized analysis."""

        response = llm.invoke(prompt)
        batch_results.append(response.content)

    # If only one batch, use its result directly (no need to combine)
    if total_batches == 1:
        return {
            "batch_analyses": batch_results,
            "analysis": batch_results[0],
        }

    # If multiple batches, do a final LLM call to combine them
    print(f"\nCombining {total_batches} batch analyses into a final summary...")

    combined_text = ""
    for i, result in enumerate(batch_results, start=1):
        combined_text += f"\n--- Batch {i} Analysis ---\n{result}\n"

    combine_prompt = f"""You are a senior DevOps engineer. Below are separate analyses of Apache error logs
that were processed in {total_batches} batches. Please combine them into ONE cohesive final analysis.

Merge duplicate error categories, add up the counts, and provide a single unified report with:

1. **Error Summary**: Combined table with total counts across all batches
2. **Root Cause Analysis**: Merged analysis for each error category
3. **Severity Assessment**: Overall severity for each category
4. **Recommended Fixes**: Consolidated, actionable steps
5. **Priority Order**: Final priority ranking based on all errors

Here are the individual batch analyses:

{combined_text}

Provide a clear, well-organized COMBINED analysis."""

    final_response = llm.invoke(combine_prompt)

    return {
        "batch_analyses": batch_results,
        "analysis": final_response.content,
    }


# ------------------------------------------------------------
# STEP 4: Build the LangGraph Graph
# ------------------------------------------------------------
# Now we connect everything together!
#
# A LangGraph "graph" defines the flow of your application:
#   - Which nodes exist
#   - In what order they run
#
# Our flow is simple and linear:
#   START --> parse_logs --> analyze_errors --> END
#
# START and END are special markers provided by LangGraph.
# ------------------------------------------------------------

from langgraph.graph import StateGraph, START, END


def build_graph():
    """Build and compile the LangGraph workflow."""

    # 1. Create a new graph using our state type
    #    This tells LangGraph what data shape to expect
    graph_builder = StateGraph(LogAnalysisState)

    # 2. Add our two nodes (functions) to the graph
    #    First argument = name (string), second = the function
    graph_builder.add_node("parse_logs", parse_logs)
    graph_builder.add_node("analyze_errors", analyze_errors)

    # 3. Connect the nodes with edges (define the flow)
    #    START -> parse_logs -> analyze_errors -> END
    graph_builder.add_edge(START, "parse_logs")
    graph_builder.add_edge("parse_logs", "analyze_errors")
    graph_builder.add_edge("analyze_errors", END)

    # 4. Compile the graph into something we can run
    #    Think of this like "building" the workflow
    graph = graph_builder.compile()

    return graph


# ------------------------------------------------------------
# STEP 5: Main Entry Point
# ------------------------------------------------------------
# This is where everything comes together. We:
#   1. Check that the OpenAI API key is set
#   2. Find the log file
#   3. Build and run the graph
#   4. Print the results
# ------------------------------------------------------------

import os
import sys


def main():
    """Run the Apache log analysis workflow."""

    # Check that the user has set their OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Please set your OpenAI API key first!")
        print()
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        print()
        print("Then run this script again.")
        sys.exit(1)

    # Find the sample log file (in the same folder as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "sample_apache.log")

    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found at {log_file}")
        sys.exit(1)

    # Print a nice header
    print("=" * 60)
    print("  Apache Log Analyzer")
    print("  Powered by LangGraph + OpenAI GPT")
    print("=" * 60)

    # Build the graph
    graph = build_graph()

    # Run the graph!
    # We pass in the initial state with:
    #   - log_file_path: where to find the log file
    #   - error_lines: empty list (the parser will fill this)
    #   - analysis: empty string (the LLM will fill this)
    result = graph.invoke({
        "log_file_path": log_file,
        "error_lines": [],
        "batch_analyses": [],
        "analysis": "",
    })

    # Print the final analysis from GPT
    print("\n" + "=" * 60)
    print("  ANALYSIS RESULT")
    print("=" * 60 + "\n")
    print(result["analysis"])
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


# This runs main() when you execute: python app.py
if __name__ == "__main__":
    main()
