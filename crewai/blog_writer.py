"""
Blog Writer CrewAI Pipeline (Colab + OpenAI)
--------------------------------------------
Given a topic, three AI agents run in sequence: (1) Planner researches and
produces a content strategy and outline; (2) Writer drafts the blog post in
markdown; (3) Editor polishes it for clarity and brand voice. 
"""

import os
from google.colab import userdata
from crewai import Agent, Task, Crew

# --- Configuration: load OpenAI API key from Colab secrets ---
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
print("OPENAI_API_KEY has been loaded from Colab secrets.")

# --- Agents: specialized roles in the content pipeline ---
# Planner: researches and creates the content strategy/outline
planner = Agent(
    role="Strategic Content Planner",
    goal="Develop comprehensive and accurate content strategies for {topic}",
    backstory="You are an expert in devising content blueprints. Your mission is to research and organize compelling and factual information on the given topic: {topic}. This structured information will then serve as the foundation for our Content Writer to craft an insightful article, ensuring the audience gains valuable knowledge and can make informed decisions.",
    allow_delegation=False,
    verbose=True
)

# Writer: turns the strategy into a full article
writer = Agent(
    role="Content Writer",
    goal="Write engaging and well-researched articles based on the content strategy provided by the planner.",
    backstory="You are a skilled and experienced content writer, capable of transforming structured information into compelling articles. Your goal is to articulate the planner's strategy into a coherent, informative, and engaging piece for the target audience.",
    allow_delegation=False,
    verbose=True
)

# Editor: reviews and polishes the draft for quality and brand voice
editor = Agent(
    role="Content Editor and Quality Control",
    goal="Refine and enhance written content for clarity, accuracy, and adherence to organizational style guidelines.",
    backstory="You are a meticulous and experienced editor responsible for the final polish of all written materials. Your task involves reviewing content submitted by the Content Writer, ensuring it meets high journalistic standards, presents information objectively, and aligns with the company's voice. You are also adept at identifying and mitigating any potentially sensitive or controversial statements to maintain a neutral and professional tone.",
    allow_delegation=False,
    verbose=True
)

# --- Tasks: ordered steps executed by the crew ---
# Task 1: Planner produces a content strategy (outline, audience, SEO, sources)
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

# Task 2: Writer drafts the blog post from the plan (markdown, SEO, structure)
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

# Task 3: Editor refines the draft (grammar, readability, brand voice)
edit = Task(
    description=("Carefully review the drafted blog post to correct any grammatical errors, improve readability, and ensure alignment with the established brand voice."),
    expected_output="A polished, ready-to-publish blog post in markdown format, free of errors and consistent with brand guidelines.",
    agent=editor
)

# --- Crew: assemble agents and tasks; execution order is plan -> write -> edit ---
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

# --- Run the crew: topic is passed as input to all agents/tasks ---
topic = "What is Crew AI"
result = crew.kickoff(inputs={"topic": topic})
