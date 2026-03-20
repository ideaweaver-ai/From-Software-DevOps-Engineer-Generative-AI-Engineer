import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- LangSmith tracing via OpenTelemetry ---
from langsmith.integrations.otel import OtelSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

current_provider = trace.get_tracer_provider()
if isinstance(current_provider, TracerProvider):
    tracer_provider = current_provider
else:
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)

tracer_provider.add_span_processor(OtelSpanProcessor())

CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
# --- End tracing setup ---

from crewai import Agent, Task, Crew

planner = Agent(
    role="Strategic Content Planner",
    goal="Develop comprehensive and accurate content strategies for {topic}",
    backstory="You are an expert in devising content blueprints. Your mission is to research and organize compelling and factual information on the given topic: {topic}. This structured information will then serve as the foundation for our Content Writer to craft an insightful article, ensuring the audience gains valuable knowledge and can make informed decisions.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging and well-researched articles based on the content strategy provided by the planner.",
    backstory="You are a skilled and experienced content writer, capable of transforming structured information into compelling articles. Your goal is to articulate the planner's strategy into a coherent, informative, and engaging piece for the target audience.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Content Editor and Quality Control",
    goal="Refine and enhance written content for clarity, accuracy, and adherence to organizational style guidelines.",
    backstory="You are a meticulous and experienced editor responsible for the final polish of all written materials. Your task involves reviewing content submitted by the Content Writer, ensuring it meets high journalistic standards, presents information objectively, and aligns with the company's voice. You are also adept at identifying and mitigating any potentially sensitive or controversial statements to maintain a neutral and professional tone.",
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "1. Analyze the current landscape to identify significant trends, key influencers, and breaking news related to {topic}.\n"
        "2. Determine the ideal readership, considering their demographics, interests, and potential questions.\n"
        "3. Construct a detailed content blueprint, including an introduction, core arguments, supporting details, and a clear call to action.\n"
        "4. Integrate relevant SEO keywords and cite credible data or source materials."
    ),
    expected_output="A thorough content strategy document comprising a detailed outline, target audience analysis, a list of SEO keywords, and compiled research resources.",
    agent=planner
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
    agent=writer
)

edit = Task(
    description="Carefully review the drafted blog post to correct any grammatical errors, improve readability, and ensure alignment with the established brand voice.",
    expected_output="A polished, ready-to-publish blog post in markdown format, free of errors and consistent with brand guidelines.",
    agent=editor
)

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

topic = "What is Crew AI"
result = crew.kickoff(inputs={"topic": topic})
print(result)
