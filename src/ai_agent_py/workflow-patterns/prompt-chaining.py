from typing import Optional
from datetime import datetime
from unittest.mock import Base
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"


class EventExtraction(BaseModel):
    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text descript a calendar event")
    confidence_score: float = Field(
        description="Confience score between 0 and 1")


class EventDetails(BaseModel):
    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event, use ISO 8601 to format this value")
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):

    confirmation_message: str = Field(
        description="Natural language confirmation message")
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable")


def extract_event_info(user_input: str) -> Optional[EventExtraction]:
    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=EventExtraction,
    )
    result = completion.choices[0].message.parsed
    if result:
        logger.info(
            f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
        )
        return result
    return None

######
# user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."
# res = extract_event_info(user_input)


def parse_event_details(description: str) -> Optional[EventDetails]:
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.",
            },
            {"role": "user", "content": description},
        ],
        response_format=EventDetails,
    )
    result = completion.choices[0].message.parsed
    if result:
        logger.info(
            f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
        )
        logger.debug(f"Participants: {', '.join(result.participants)}")
        return result
    return None


###############
# description = "Schedule a 1-hour team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."
# event_details = parse_event_details(description)


def generate_confirmation(event_details: EventDetails) -> Optional[EventConfirmation]:
    logger.info("Generating confirmation message")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation message for the event. Sign of with your name; Susie",
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_format=EventConfirmation,
    )
    result = completion.choices[0].message.parsed
    logger.info("Confirmation message generated successfully")
    return result


def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_event_info(user_input)

    if not initial_extraction:
        logger.warning("Gate check failed - no extraction result")
        return None

    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    # Second LLM call: Get detailed event information
    event_details = parse_event_details(initial_extraction.description)
    if not event_details:
        logger.warning("Failed to parse event details")
        return None

    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details)
    logger.info("Calendar request processing completed successfully")
    return confirmation

# Testing with user input
user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")
