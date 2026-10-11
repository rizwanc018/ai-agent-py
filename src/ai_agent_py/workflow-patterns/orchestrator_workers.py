import os
from openai import OpenAI
from pydantic import BaseModel, Field

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SubTask(BaseModel):
    section_type: str = Field(description="Type of blog section to write.")
    description: str = Field(description="What this section should cover.")
    style_guide: str = Field(description="Writing style for this section")
    target_length: str = Field(
        description=" Target word count for this section")


class OrchestratorPlan(BaseModel):
    topic_analysis: str = Field(description="Analysis of blog topic.")
    target_audience: str = Field(description="Intended audience for the blog.")
    sections: list[SubTask] = Field(description="List of sections to write.")


class SectionContent(BaseModel):
    content: str = Field(description="Written content for the section")
    key_points: list[str] = Field(description="Main points covered")


class SuggestedEdits(BaseModel):
    section_name: str = Field(description="Name of section")
    suggested_edit: str = Field(description="Suggested edit")


class ReviewFeedback(BaseModel):
    cohesion_score: float = Field(
        description="How well sections flow together (0-1)")
    suggested_edits: list[SuggestedEdits] = Field(
        description="Suggested edits by section"
    )
    final_version: str = Field(description="Complete, polished blog post")


ORCHESTRATOR_PROMPT = """
Analyze this blog topic and break it down into logical sections.

Topic: {topic}
Target Length: {target_length} words
Style: {style}

Return your response in this format:

# Analysis
Analyze the topic and explain how it should be structured.
Consider the narrative flow and how sections will work together.

# Target Audience
Define the target audience and their interests/needs.

# Sections
## Section 1
- Type: section_type
- Description: what this section should cover
- Style: writing style guidelines

[Additional sections as needed...]
"""

WORKER_PROMPT = """
Write a blog section based on:
Topic: {topic}
Section Type: {section_type}
Section Goal: {description}
Style Guide: {style_guide}

Return your response in this format:

# Content
[Your section content here, following the style guide]

# Key Points
- Main point 1
- Main point 2
[Additional points as needed...]
"""

REVIEWER_PROMPT = """
Review this blog post for cohesion and flow:

Topic: {topic}
Target Audience: {audience}

Sections:
{sections}

Provide a cohesion score between 0.0 and 1.0, suggested edits for each section if needed, and a final polished version of the complete post.

The cohesion score should reflect how well the sections flow together, with 1.0 being perfect cohesion.
For suggested edits, focus on improving transitions and maintaining consistent tone across sections.
The final version should incorporate your suggested improvements into a polished, cohesive blog post.
"""


def write_section(topic: str, section: SubTask):
    logger.info(f"Creating section about: {section.section_type}")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": WORKER_PROMPT.format(
                        topic=topic,
                        section_type=section.section_type,
                        description=section.description,
                        style_guide=section.style_guide,
                        target_length=section.target_length,
                ),
            }
        ],
        response_format=SectionContent,
    )

    message = completion.choices[0].message
    if message.parsed is None:
        error_msg = f"Failed to write about {section.section_type}. Refusal: {message.refusal}"
        logger.info(error_msg)
        raise ValueError(error_msg)
    result = message.parsed
    logger.info(
        f"Completed - {section.section_type}")
    return result


def get_plan(topic: str, target_length: int, style: str) -> OrchestratorPlan:
    logger.info("Creating plan")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": ORCHESTRATOR_PROMPT.format(
                        topic=topic, target_length=target_length, style=style
                )
            },

        ],
        response_format=OrchestratorPlan
    )

    message = completion.choices[0].message
    if message.parsed is None:
        error_msg = f"Failed to orchestrate plan. Refusal: {message.refusal}"
        logger.info(error_msg)
        raise ValueError(error_msg)
    result = message.parsed
    logger.info(
        f"Plan Orchestrated - Topic-analysis: {result.topic_analysis},\nTraget-audience: {result.target_audience},\nSections: {result.sections}")
    return result


def review_blog(topic: str, plan: OrchestratorPlan, blog_dict: dict[str, str]) -> ReviewFeedback:
    logger.info("Reviewing blog")

    sections_text = "\n\n".join(
        f"## {section_type}\n{content}"
        for section_type, content in blog_dict.items()
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": REVIEWER_PROMPT.format(
                    topic=topic,
                    audience=plan.target_audience,
                    sections=sections_text,
                ),
            }
        ],
        response_format=ReviewFeedback,
    )

    message = completion.choices[0].message
    if message.parsed is None:
        error_msg = f"Failed to review blog. Refusal: {message.refusal}"
        logger.info(error_msg)
        raise ValueError(error_msg)
    result = message.parsed
    logger.info(f"Review complete - Cohesion score: {result.cohesion_score}")
    return result


topic = "The impact of AI on software development"
target_length = 1200
style = "technical but accessible"

plan = get_plan(topic, target_length, style)
print(plan.model_dump_json(indent=2))

blog_dict = {}

for i, section in enumerate(plan.sections):
    content = write_section(topic, section)
    blog_dict[f"{i+1}. {section.section_type}"] = content.content
    
print(blog_dict)

review = review_blog(topic, plan, blog_dict)
# print(review.model_dump_json(indent=2))
print(f"\n>> Cohesion score = {review.cohesion_score}\n")
print(review.final_version)
