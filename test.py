#!/usr/bin/env python3

import os
import sys
import time

from dotenv import load_dotenv

from openai import OpenAI


_ = load_dotenv()


BASE_URL = os.environ.get("BASE_URL", "http://localhost:4000/v1")
API_KEY = "dummy-api-key"
MODEL = "x-gpt-5"
HEAVY_PROMPT = (
    "What are the latest developments in quantum computing in 2024? "
    "Search the web and provide a comprehensive analysis including key "
    "breakthroughs, major companies involved, and potential applications."
)
MIN_REASONING_DURATION = 90
MAX_COMPLETION_TOKENS = 128000
RESPONSE_PREFIX = ":test:"
HEARTBEAT_MARKER = "."

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def test_chat_stream_direct() -> None:
    print("chat.completions stream=true, reasoning_effort=high, tools=[web_search]")

    # Benchmark direct OpenAI first
    print("Direct OpenAI...")
    direct_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_STREAMING"))

    start_time = time.time()
    stream = direct_client.responses.create(
        model="gpt-5",
        instructions="Reason carefully. Use rigorous structure.",
        input=HEAVY_PROMPT,
        tools=[{"type": "web_search"}],
        reasoning={"effort": "high"},
        stream=True,
    )

    for _ in stream:
        pass

    direct_duration = time.time() - start_time

    print(f"Direct OpenAI: {direct_duration:.2f}s")


def test_chat_stream_proxy() -> None:
    print("chat.completions stream=true, reasoning_effort=high, tools=[web_search]")

    print("Proxy...")

    start_time = time.time()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Reason carefully. Use rigorous structure.",
            },
            {"role": "user", "content": HEAVY_PROMPT},
        ],
        stream=True,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort="high",
    )

    for chunk in stream:
        _ = sys.stdout.write(chunk.choices[0].delta.content or "")
        _ = sys.stdout.flush()

    proxy_duration = time.time() - start_time

    print(f"\nProxy Duration: {proxy_duration:.2f}s")


def test_conversation() -> None:
    print("chat.completions conversation")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a chat bot.",
            },
            {"role": "user", "content": "Hi! I'm Alex."},
        ],
        stream=True,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    assistant = ""

    for chunk in stream:
        assistant += chunk.choices[0].delta.content or ""
        _ = sys.stdout.write(chunk.choices[0].delta.content or "")
        _ = sys.stdout.flush()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a chat bot.",
            },
            {"role": "user", "content": "Hi! I'm Alex."},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "What's my name?"},
        ],
        stream=True,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    for chunk in stream:
        _ = sys.stdout.write(chunk.choices[0].delta.content or "")
        _ = sys.stdout.flush()


def test_deep_research() -> None:
    print("openai deep research")

    stream = client.chat.completions.create(
        model="x-o3-deep-research",
        messages=[
            {"role": "user", "content": "Карточка компании Лукойл"},
        ],
        stream=True,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    for chunk in stream:
        _ = sys.stdout.write(chunk.choices[0].delta.content or "")
        _ = sys.stdout.flush()


def test_perplexity_stream(model: str) -> None:
    print("Perplexity streaming...")

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "What are the latest AI developments in 2024?"}
        ],
        stream=True,
        max_tokens=4096,
    )

    for chunk in stream:
        _ = sys.stdout.write(chunk.choices[0].delta.content or "")
        _ = sys.stdout.flush()


if __name__ == "__main__":
    # test_chat_stream_direct()
    test_chat_stream_proxy()
    # test_conversation()
    # test_deep_research()
    # test_perplexity_stream("x-sonar-pro")
    # test_perplexity_stream("x-sonar-reasoning-pro-high")
