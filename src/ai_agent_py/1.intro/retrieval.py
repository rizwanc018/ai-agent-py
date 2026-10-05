import json
import os
from unittest import result

from openai import OpenAI
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessageFunctionToolCall, ChatCompletionMessageParam
from openai.types.chat import ChatCompletionToolParam

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def search_kb(question: str):
    with open("kb.json", "r") as f:
        return json.load(f)


tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

completion.model_dump()

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


def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


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
class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")


completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=KBResponse,
)

final_response = completion_2.choices[0].message.parsed
if final_response:
    final_response.answer
    final_response.source


messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

completion_3 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

completion_3.choices[0].message.content