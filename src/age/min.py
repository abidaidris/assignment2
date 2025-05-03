import os
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_default_openai_client(external_client)
set_tracing_disabled(True)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)


def my_first_agent():
    agent = Agent(
        name="Assistant", instructions="You are a helpful assistant", model=model
    )
    result = Runner.run_sync(agent, "Hello, world!")
    print(result.final_output)
