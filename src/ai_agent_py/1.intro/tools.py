import json
import os
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessageCustomToolCall, ChatCompletionMessageParam
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from traitlets import Instance


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


tools: list[ChatCompletionToolParam] = [

    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful weather assistant."

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Paris today?"},
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

completion.model_dump()


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)


assistant_message = completion.choices[0].message

messages.append({
    "role": "assistant",
    "content": assistant_message.content,
    "tool_calls": [
        {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }, }
        for tool_call in assistant_message.tool_calls or []
        if isinstance(tool_call, ChatCompletionMessageFunctionToolCall)

    ],
})

for tool_call in completion.choices[0].message.tool_calls or []:
    if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
        continue

    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id,
            "content": json.dumps(result)}
    )


class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=WeatherResponse,
)

final_response = completion_2.choices[0].message.parsed
if final_response: 
    final_response.temperature
    final_response.response