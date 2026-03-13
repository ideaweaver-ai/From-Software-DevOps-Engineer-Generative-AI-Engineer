"""
Market Research CrewAI Pipeline (Codio / OpenAI)
------------------------------------------------

What we are doing:
  We run a market research pipeline in CrewAI: you give it an AI product idea
  (and optionally an image). Five agents run in order. The Market Researcher
  sizes the market and trends (and can describe an input image via a vision tool).
  The Competitive Analyst maps competitors and gaps and can create comparison
  charts. The Customer Researcher defines segments, pain points, and willingness
  to pay. The Product Strategist proposes MVP, differentiation, and roadmap and
  can generate a strategy diagram. The Business Analyst pulls everything into
  one report with pricing, revenue model, risks, and a go/no-go recommendation,
  and uses create_chart and generate_image (GPT-5.4 via the Responses API) so
  the report includes data charts and AI-generated visuals. Research agents use
  a stronger LLM (e.g. gpt-4o) and synthesis agents a lighter one (e.g. gpt-4o-mini).
  Set up for Codio: set OPENAI_API_KEY in Codio environment variables or in a
  .env file in the project root, then run this script. Optional: set
  LANGSMITH_API_KEY and LANGSMITH_PROJECT to push CrewAI traces to LangSmith.
"""

import base64
import os
from pathlib import Path

# Load .env so OPENAI_API_KEY and LANGSMITH_* are available (Codio: or set in env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Optional: push CrewAI traces to LangSmith (set LANGSMITH_API_KEY and LANGSMITH_PROJECT) ---
try:
    if os.environ.get("LANGSMITH_API_KEY"):
        from langsmith.integrations.otel import OtelSpanProcessor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        current_provider = trace.get_tracer_provider()
        tracer_provider = current_provider if isinstance(current_provider, TracerProvider) else TracerProvider()
        if not isinstance(current_provider, TracerProvider):
            trace.set_tracer_provider(tracer_provider)
        tracer_provider.add_span_processor(OtelSpanProcessor())
        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        print("LangSmith tracing enabled (traces will appear in LANGSMITH_PROJECT).")
except Exception as e:
    print(f"LangSmith tracing skipped: {e}")

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

# --- Configuration: OpenAI API key (Codio env vars or .env) ---
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. In Codio: set it in Environment Variables, or add it to .env.")
else:
    print("OPENAI_API_KEY loaded.")

# --- LLMs: use different models per agent (research vs synthesis) ---
# Research agents use a stronger model; synthesis agents use a faster/cheaper one.
llm_research = LLM(model="gpt-4o", temperature=0.3)
llm_synthesis = LLM(model="gpt-4o-mini", temperature=0.2)


# --- Tools: image description (vision) and chart creation ---
@tool("describe_image")
def describe_image(image_path: str = "", image_url: str = "") -> str:
    """
    Describe the contents of an image for market research. Use when the user
    provides image_path (file path) or image_url (URL). Returns a text summary
    of what is in the image so you can use it in your analysis. If both are
    empty, return a short message that no image was provided.
    """
    if not image_path and not image_url:
        return "No image was provided. Proceed with your analysis without image context."
    try:
        from openai import OpenAI
        client = OpenAI()
        content = [{"type": "text", "text": "Describe this image in detail for market or product research. Include any text, numbers, charts, or diagrams you see."}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            path = Path(image_path)
            if not path.exists():
                return f"Image file not found: {image_path}"
            with open(path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            ext = path.suffix.lower() or ".png"
            mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        r = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": content}], max_tokens=1024)
        return r.choices[0].message.content or "Could not describe image."
    except Exception as e:
        return f"Error describing image: {e!s}"


@tool("create_chart")
def create_chart(title: str, labels: str, values: str, output_path: str = "research_output/chart.png") -> str:
    """
    Create a bar chart and save it to a file. Use for competitor comparison,
    market share, or revenue projections. Arguments: title (chart title),
    labels (comma-separated, e.g. 'A,B,C'), values (comma-separated numbers,
    e.g. '30,25,45'), output_path (optional, default research_output/chart.png).
    Returns the path where the chart was saved or an error message.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        label_list = [x.strip() for x in labels.split(",") if x.strip()]
        value_list = [float(x.strip().replace(",", "")) for x in values.split(",") if x.strip()]
        if len(label_list) != len(value_list) or not label_list:
            return "Labels and values must be the same length and non-empty (comma-separated)."
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ax.bar(label_list, value_list, color="steelblue", edgecolor="navy")
        ax.set_title(title)
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out, dpi=100)
        plt.close()
        return f"Chart saved to {out.absolute()}. Use this path in your report."
    except Exception as e:
        return f"Error creating chart: {e!s}"


# Model for image generation: GPT-5.4 handles prompt reasoning and image gen via Responses API
IMAGE_GEN_MODEL = "gpt-5.4"


@tool("generate_image")
def generate_image(prompt: str, filename: str = "research_output/generated_image.png") -> str:
    """
    Generate an image from a text description using GPT-5.4 (Responses API with image_generation tool).
    The model interprets your prompt and generates the image, so reasoning and visuals stay aligned.
    Use for custom visuals (e.g. conceptual diagram, infographic, roadmap, positioning graphic).
    Arguments: prompt (detailed English description; be specific about style);
    filename (optional path to save). Returns the saved path or an error message.
    """
    if not prompt or not prompt.strip():
        return "Please provide a non-empty prompt describing the image to generate."
    try:
        from openai import OpenAI
        client = OpenAI()
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        full_prompt = (
            f"Generate an image for a business report: {prompt.strip()} "
            "Style: clean, professional. Minimal or no text in the image."
        )
        r = client.responses.create(
            model=IMAGE_GEN_MODEL,
            input=full_prompt[:4000],
            tools=[{"type": "image_generation", "action": "generate"}],
        )
        image_b64 = None
        for item in getattr(r, "output", []) or []:
            if getattr(item, "type", None) == "image_generation_call":
                image_b64 = getattr(item, "result", None)
                break
        if not image_b64:
            return "No image in response; the model may have declined or the API format changed."
        out.write_bytes(base64.standard_b64decode(image_b64))
        return f"Image saved to {out.absolute()}. Reference in your report, e.g. ![Description]({out})"
    except Exception as e:
        return f"Error generating image: {e!s}"


# --- Agents: each has explicit LLM; research agents get research LLM, synthesis get synthesis LLM ---
# Market Researcher: sizes the market and spots trends; can use describe_image
market_researcher = Agent(
    role="Market Research Specialist",
    goal="Run solid market analysis for {product_idea}: market size, growth, industry shifts, and how fast the space is adopting new tech.",
    backstory="You are a senior market analyst focused on AI and tech. You have long experience sizing markets, reading industry dynamics, and spotting adoption patterns. You turn data and trends into clear, actionable market views and have helped many teams decide where to play and when.",
    allow_delegation=False,
    verbose=True,
    llm=llm_research,
    tools=[describe_image],
)

# Competitive Analyst: maps rivals and finds white space; can create comparison charts
competitive_analyst = Agent(
    role="Competitive Intelligence Analyst",
    goal="List and assess direct and indirect competitors for {product_idea}: what they do, strengths and weaknesses, and how they're positioned, so we can see gaps and options.",
    backstory="You are a competitive intelligence lead in AI and SaaS. You are used to building competitor maps, comparing offerings and pricing, and understanding go-to-market moves. You help teams find a clear position and spot openings in crowded markets.",
    allow_delegation=False,
    verbose=True,
    llm=llm_research,
    tools=[create_chart],
)

# Customer Researcher: who they are and what they'd pay for
customer_researcher = Agent(
    role="Customer Insights Researcher",
    goal="Build a clear picture of who would buy {product_idea}: segments, pain points, needs, behavior, and willingness to pay.",
    backstory="You are a customer research lead in B2B and B2C tech. You define personas, run needs and jobs-to-be-done style analysis, and identify where and how to reach customers. You have helped many AI products sharpen product–market fit and messaging.",
    allow_delegation=False,
    verbose=True,
    llm=llm_research,
)

# Product Strategist: MVP, differentiation, and roadmap; can generate a strategy diagram
product_strategist = Agent(
    role="Product Strategy Advisor",
    goal="Propose a product strategy for {product_idea}: MVP scope, differentiation, technical feasibility, and a phased roadmap using the market, competitive, and customer inputs.",
    backstory="You are an experienced product leader who has shipped multiple AI products. You are strong at prioritisation (e.g. RICE, Kano), balancing business and technical constraints, and turning research into concrete product choices and roadmaps.",
    allow_delegation=False,
    verbose=True,
    llm=llm_synthesis,
    tools=[generate_image],
)

# Business Analyst: final report; can create charts and LLM-generated images
business_analyst = Agent(
    role="Business Analyst and Report Synthesizer",
    goal="Turn all research into one business report for {product_idea}: pricing approach, revenue model, main risks, and a clear go/no-go recommendation.",
    backstory="You are a senior business analyst with a strong background in strategy and financial modelling for tech and AI. You build pricing and revenue models, stress-test assumptions, and write concise executive summaries that support real decisions.",
    allow_delegation=False,
    verbose=True,
    llm=llm_synthesis,
    tools=[create_chart, generate_image],
)

# --- Tasks: ordered steps; image at stage 1, charts at stage 2 and 5 ---
# Task 1: Market Researcher; if image_path or image_url provided, use describe_image first
market_task = Task(
    description=(
        "1. If image_path or image_url was provided, call describe_image with that value (image_path='{image_path}' or image_url='{image_url}') and weave the result into your analysis.\n"
        "2. Define the relevant market for the product idea: {product_idea} and estimate its size (TAM/SAM/SOM style).\n"
        "3. Summarise growth trends, drivers, and where the industry is heading.\n"
        "4. Describe how quickly the space adopts new technology and any regulatory or structural factors.\n"
        "5. Note data sources and assumptions so others can follow the logic."
    ),
    expected_output="A concise market analysis: size, growth, dynamics, adoption patterns, and key assumptions with sources. If an image was analyzed, include its relevance.",
    agent=market_researcher,
)

# Task 2: Competitive Analyst; may create a comparison chart
competitive_task = Task(
    description=(
        "1. List direct and indirect competitors for {product_idea} and briefly describe what each offers.\n"
        "2. For each, summarise strengths, weaknesses, and how they are positioned (e.g. premium, volume, niche).\n"
        "3. Identify gaps and opportunities where the product could stand out.\n"
        "4. Optionally use the create_chart tool to create a competitor comparison chart (e.g. market share or feature comparison) and mention the saved chart path in your output.\n"
        "5. Keep the analysis factual and actionable."
    ),
    expected_output="A competitive overview: competitor list, comparison of offerings and positioning, and identified gaps and opportunities. Optionally a chart path if you created one.",
    agent=competitive_analyst,
)

# Task 3: Customer Researcher delivers segments and willingness to pay
customer_task = Task(
    description=(
        "1. Define 2–3 target customer segments for {product_idea} with clear criteria (e.g. firm size, role, industry).\n"
        "2. For each segment, describe main pain points, needs, and how they behave when evaluating solutions.\n"
        "3. Indicate willingness to pay and how they typically discover and buy similar products.\n"
        "4. Suggest how to reach them (channels, messaging angles)."
    ),
    expected_output="A customer insights summary: segments, pains, needs, behaviour, willingness to pay, and acquisition angles.",
    agent=customer_researcher,
)

# Task 4: Product Strategist delivers MVP and roadmap; may generate a strategy visual
strategy_task = Task(
    description=(
        "1. Using the market, competitive, and customer inputs, propose an MVP for {product_idea}: must-have features and what to defer.\n"
        "2. State how the product should differentiate and why that fits the gaps and segments.\n"
        "3. Comment on technical feasibility and main build vs buy choices.\n"
        "4. Outline a simple phased roadmap (e.g. MVP → v1 → scale) with milestones.\n"
        "5. Optionally use generate_image to create a single diagram (e.g. product roadmap timeline, or positioning vs competitors) and give the saved path in your output so the report can reference it."
    ),
    expected_output="A product strategy memo: MVP scope, differentiation, feasibility, and a phased roadmap with milestones. Optionally the path to one generated diagram.",
    agent=product_strategist,
    context=[market_task, competitive_task, customer_task],
)

# Task 5: Business Analyst; create charts and LLM-generated images so the report looks authentic
report_task = Task(
    description=(
        "1. Synthesise the market, competitive, customer, and product strategy work into one business report for {product_idea}.\n"
        "2. Propose a pricing strategy and a simple revenue model (e.g. subscription, usage, one-time) with brief rationale.\n"
        "3. List the main risks (market, competition, execution, regulation) and how they might be mitigated.\n"
        "4. Use create_chart for at least one data-driven chart (e.g. market size, revenue projection, or competitor comparison) and reference the saved path in the report.\n"
        "5. Use generate_image to create 1–2 custom visuals so the report does not look generic: e.g. a conceptual diagram (product positioning, value chain), a summary infographic, or a simple roadmap illustration. Pass a clear, detailed prompt describing what to draw; save each with a distinct filename (e.g. research_output/positioning.png, research_output/roadmap.png). Reference each generated image in the report with markdown, e.g. ![Alt](path).\n"
        "6. End with a clear go/no-go recommendation and 2–3 next steps."
    ),
    expected_output="A final business report in markdown: executive summary, pricing and revenue model, risk analysis, and go/no-go recommendation with next steps. Include at least one chart and 1–2 LLM-generated images, each referenced in the report so the output looks authentic and professional.",
    agent=business_analyst,
    context=[market_task, competitive_task, customer_task, strategy_task],
)

# --- Crew: agents and tasks in order; later tasks use context from earlier ones ---
crew = Crew(
    agents=[market_researcher, competitive_analyst, customer_researcher, product_strategist, business_analyst],
    tasks=[market_task, competitive_task, customer_task, strategy_task, report_task],
    verbose=True,
)

# --- Run the crew: product_idea required; image_path and image_url optional ---
if __name__ == "__main__":
    product_idea = "An AI-powered tool that helps DevOps teams automate incident response and runbook execution"
    inputs = {
        "product_idea": product_idea,
        "image_path": "",   # optional: local path to image for market research
        "image_url": "",     # optional: URL of image for market research
    }
    # To add an image at the start, set one of:
    # inputs["image_path"] = "path/to/screenshot.png"
    # inputs["image_url"] = "https://example.com/diagram.png"
    result = crew.kickoff(inputs=inputs)
    print("\n--- Final report ---\n")
    print(result)
