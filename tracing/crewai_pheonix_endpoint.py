"""
CrewAI blog crew (same flow as 5_crewai_blog.py) with traces sent to Phoenix Arize Cloud.

Env (see .env):
  - OPENAI_API_KEY
  - PHOENIX_COLLECTOR_ENDPOINT  e.g. https://app.phoenix.arize.com/s/<space_id>
  - PHOENIX_API_KEY             (if your space requires it)
  - PHOENIX_PROJECT_NAME        optional; defaults below

Also install:
  pip install opentelemetry-instrumentation-crewai opentelemetry-instrumentation-openai
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# If a corporate proxy blocks Phoenix Cloud, bypass it for this process (optional).
for _k in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(_k, None)
_phoenix_host = "app.phoenix.arize.com"
_no_proxy = os.getenv("NO_PROXY", "")
if _phoenix_host not in _no_proxy:
    os.environ["NO_PROXY"] = (
        (_no_proxy + "," if _no_proxy else "") + _phoenix_host
    ).strip(",")


def _cloud_traces_endpoint() -> str:
    """
    OTLP/HTTP traces URL for Phoenix Cloud:
      https://app.phoenix.arize.com/s/<space_id>/v1/traces
    """
    if explicit := os.getenv("PHOENIX_HTTP_TRACES_ENDPOINT"):
        return explicit.rstrip("/")

    if collector := os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        collector = collector.rstrip("/")
        if collector.endswith("/v1/traces"):
            return collector
        return collector + "/v1/traces"

    base = os.getenv("PHOENIX_CLOUD_BASE_URL")
    if not base:
        raise RuntimeError(
            "Set PHOENIX_COLLECTOR_ENDPOINT or PHOENIX_HTTP_TRACES_ENDPOINT or "
            "PHOENIX_CLOUD_BASE_URL for Phoenix Cloud."
        )
    return base.rstrip("/") + "/v1/traces"


# --- Phoenix Cloud OTEL export + CrewAI / OpenAI spans ---
from opentelemetry import trace
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from phoenix.otel import register

_endpoint = _cloud_traces_endpoint()
_project = os.getenv("PHOENIX_PROJECT_NAME", "crewai-blog-phoenix-cloud")

print(f"Phoenix project: {_project}")
print(f"Phoenix traces endpoint: {_endpoint}")

tracer_provider = register(
    project_name=_project,
    auto_instrument=False,
    endpoint=_endpoint,
    protocol="http/protobuf",
    api_key=os.getenv("PHOENIX_API_KEY"),
    verbose=False,
)

# register() should return an SDK TracerProvider; normalize if needed
if not isinstance(tracer_provider, TracerProvider):
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        tracer_provider = current
    else:
        raise RuntimeError(
            "Could not obtain OpenTelemetry TracerProvider after phoenix.otel.register()"
        )

CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
# --- End tracing setup ---

from crewai import Agent, Crew, Task

planner = Agent(
    role="Strategic Content Planner",
    goal="Develop comprehensive and accurate content strategies for {topic}",
    backstory="You are an expert in devising content blueprints. Your mission is to research and organize compelling and factual information on the given topic: {topic}. This structured information will then serve as the foundation for our Content Writer to craft an insightful article, ensuring the audience gains valuable knowledge and can make informed decisions.",
    allow_delegation=False,
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging and well-researched articles based on the content strategy provided by the planner.",
    backstory="You are a skilled and experienced content writer, capable of transforming structured information into compelling articles. Your goal is to articulate the planner's strategy into a coherent, informative, and engaging piece for the target audience.",
    allow_delegation=False,
    verbose=True,
)

editor = Agent(
    role="Content Editor and Quality Control",
    goal="Refine and enhance written content for clarity, accuracy, and adherence to organizational style guidelines.",
    backstory="You are a meticulous and experienced editor responsible for the final polish of all written materials. Your task involves reviewing content submitted by the Content Writer, ensuring it meets high journalistic standards, presents information objectively, and aligns with the company's voice. You are also adept at identifying and mitigating any potentially sensitive or controversial statements to maintain a neutral and professional tone.",
    allow_delegation=False,
    verbose=True,
)

plan = Task(
    description=(
        "1. Analyze the current landscape to identify significant trends, key influencers, and breaking news related to {topic}.\n"
        "2. Determine the ideal readership, considering their demographics, interests, and potential questions.\n"
        "3. Construct a detailed content blueprint, including an introduction, core arguments, supporting details, and a clear call to action.\n"
        "4. Integrate relevant SEO keywords and cite credible data or source materials."
    ),
    expected_output="A thorough content strategy document comprising a detailed outline, target audience analysis, a list of SEO keywords, and compiled research resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Utilize the approved content plan to compose a compelling and informative article on {topic}.\n"
        "2. Seamlessly weave in designated SEO keywords throughout the text.\n"
        "3. Ensure all sections and subheadings are engaging and appropriately titled.\n"
        "4. Structure the article with an captivating opening, a comprehensive main body, and a concise concluding summary.\n"
        "5. Conduct a meticulous review for grammatical correctness and consistency with the organizational tone of voice.\n"
    ),
    expected_output="A professionally written blog post formatted in markdown, prepared for publication, with each major section containing 2 to 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description="Carefully review the drafted blog post to correct any grammatical errors, improve readability, and ensure alignment with the established brand voice.",
    expected_output="A polished, ready-to-publish blog post in markdown format, free of errors and consistent with brand guidelines.",
    agent=editor,
)

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True,
)

if __name__ == "__main__":
    topic = "What is Crew AI"
    result = crew.kickoff(inputs={"topic": topic})
    print(result)
