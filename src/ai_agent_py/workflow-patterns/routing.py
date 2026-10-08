from typing import Optional, Literal
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

# Model


class CalendarRequestType(BaseModel):
    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of calendar request being made"
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 1")
    description: str = Field(description="Description of the request")


class NewEventDetails(BaseModel):
    name: str = Field(description="Name of the event")
    date: str = Field(description="Date and time of the event (ISO 8601)")
    duration_minutes: int = Field(description="Duration in minutes")
    participants: list[str] = Field(description="List of participants")


class Change(BaseModel):
    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: list[Change] = Field(description="List of changes to make")
    participants_to_add: list[str] = Field(
        description="New participants to add")
    participants_to_remove: list[str] = Field(
        description="Participants to remove")


class CalendarResponse(BaseModel):
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(
        description="Calendar link if applicable")


def decide_calendar_request_type(user_input: str) -> CalendarRequestType:
    logger.info("Deciding calendar request type")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Determine if this is a request to create a new calendar event or modify an existing one.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=CalendarRequestType,
    )

    message = completion.choices[0].message
    if message.parsed is None:
        error_msg = f"Failed to parse request type. Refusal: {message.refusal}"
        logger.info(error_msg)
        raise ValueError(error_msg)

    result = message.parsed
    logger.info(
        f"Request routed as: {result.request_type} with confidence: {result.confidence_score}"
    )
    logger.info(f" >>> Description: {result.description}")
    return result


def handle_new_event(description: str) -> CalendarResponse:
    logger.info("Processing new event request")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract details for creating a new calendar event.",
            },
            {"role": "user", "content": description},
        ],
        response_format=NewEventDetails,
    )
    message = completion.choices[0].message

    if message.parsed is None:
        error_msg = f"Failed to extract details for creating new calendar event. Refusal: {message.refusal}"
        logger.info(error_msg)
        raise ValueError(error_msg)

    details = message.parsed
    logger.info(f"New event: {details.model_dump_json(indent=2)}")

    return CalendarResponse(
        success=True,
        message=f"Created new event '{details.name}' for {details.date} with {', '.join(details.participants)}",
        calendar_link=f"calendar://new?event={details.name}",
    )


def handle_modify_event(description: str) -> CalendarResponse:
    logger.info("Processing event modification request")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract details for modifying an existing calendar event.",
            },
            {"role": "user", "content": description},
        ],
        response_format=ModifyEventDetails,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        error_msg = f"Failed to extract modification details. Refusal: {message.refusal}"
        logger.info(error_msg)
        raise ValueError(error_msg)

    details = message.parsed

    logger.info(f"Modified event: {details.model_dump_json(indent=2)}")

    return CalendarResponse(
        success=True,
        message=f"Modified event '{details.event_identifier}' with the requested changes",
        calendar_link=f"calendar://modify?event={details.event_identifier}",
    )


def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
    logger.info("Processing calendar request")

    classification = decide_calendar_request_type(user_input)

    if (classification.confidence_score < 0.7):
        logger.warning(
            f"Low confidence score: {classification.confidence_score}")
        return None

    if classification.request_type == "new_event":
        return handle_new_event(classification.description)
    elif classification.request_type == "modify_event":
        return handle_modify_event(classification.description)
    else:
        logger.warning("Request type not supported")
        return None


new_event_input = "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
result = process_calendar_request(new_event_input)
if result:
    print(f"Response: {result.message}")
    print(f"Link: {result.calendar_link}")
