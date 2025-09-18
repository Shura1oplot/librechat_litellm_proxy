#!/usr/bin/env python3

import os
import time
from collections.abc import Iterator, AsyncIterator
from typing import Any, Callable, cast
import re
import asyncio

from dotenv import load_dotenv

import httpx

from openai import AsyncOpenAI
from openai.types.responses import Response

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


_ = load_dotenv()


DEFAULT_OUTBOUND_MODEL = "gpt-5"
DEFAULT_HEARTBEAT_INTERVAL = 15.0
DEFAULT_HEARTBEAT_MARKER = "."

STREAMING_CHUNK_SIZE = 50
STREAMING_CHUNK_DELAY = 0.02

BACKGROUND_POLL_INTERVAL = 5.0

DEFAULT_SYSTEM_PROMPT = """\
# Communication style
Professional, expert, unbiased, structured, with sound judgement, \
rationally skeptical, insights- and data-driven, business value-oriented, \
concise and straightforward communication with reasonable depth, no sugar-coating, \
and no bullshit. Executive level, unless I asked for details.

# Guidelines
- Reply in the language of the query.
- Prefer tables and bullet points.
- Brief description of your logic.
- Do not provide a 90‑day plan or roadmap unless asked.

# Analytics tasks
- Figures: units, timeframes, links to sources + context summary.
- Analyze multiple sources, compare figures, explain discrepancies. \
Mention evidence conflicts if any.
- Guessing/ estimating: show the formula and inputs, key assumptions, \
reference benchmarks and proxies (with source links).
- Ensure consistency of time periods when combining figures. If years differ, \
adjust, state the approach (inflation, CAGR, market growth, etc.).
- Check math and logic; compute business metrics (e.g. unit-economics KPIs); \
benchmark vs competitors/ market/ proxies; cite sources.

# Research tasks
- Provide source links. Prefer primary sources.
- Facts are time‑sensitive. Write the year near the link.
- For software, include the last update date and exclude abandoned projects.
- Treat my list as illustrative if I write “etc.”; add 2–5 relevant examples \
within scope.
- Say insufficient evidence if anything found is irrelevant.

# Q/A
- Avoid human and AI biases.
- Explore opposite opinions as well.
- Adjust your answer as majority of people would answer.\
"""


ROUTABLE_MODELS = [
    "x-gpt-5",
    "x-gpt-5-thinking",
    "x-gpt-5-thinking-high",
    "x-o4-mini-deep-research",
    "x-o3-deep-research",
    "x-perplexity-pro",
    "x-perplexity-pro-high",
    "x-perplexity-reasoning-pro",
    "x-perplexity-reasoning-pro-high",
    "x-perplexity-deep-research",
    "x-perplexity-deep-research-high",
]

ROUTABLE_CHOICES = ", ".join(ROUTABLE_MODELS)

ROUTER_SYSTEM_PROMPT = (
    "You are a routing agent. Choose the best model for the user message. "
    "Valid options: " + ROUTABLE_CHOICES + ". Return only the model name."
)


class OpenAIResponsesBridge(CustomLLM):

    @staticmethod
    def _get_input_from_messages(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message["role"] == "user":
                return str(message["content"])

        raise ValueError(messages)

    @staticmethod
    def _get_conversation_id(messages: list[dict[str, Any]]) -> str | None:
        for message in messages:
            if message["role"] == "assistant":
                match = re.search(r"<conv_id=([^>]+)>", str(message["content"]))

                if match:
                    return match.group(1)

        return None

    @staticmethod
    def _get_instructions_from_messages(messages: list[dict[str, Any]]) -> str | None:
        for message in messages:
            if message["role"] in ("system", "developer"):
                return str(message["content"])

        return DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                if hasattr(output_item, "content") and output_item.content:
                    for content_item in output_item.content:
                        if hasattr(content_item, "text") and content_item.text:
                            return str(content_item.text)

        elif hasattr(response, "output_text"):
            return response.output_text

        return str(response)

    @staticmethod
    async def _background_responses(aclient: AsyncOpenAI, _id: str) -> Response:
        while True:
            response = await aclient.responses.retrieve(_id)

            if response.status in {"in_progress", "queued"}:
                await asyncio.sleep(BACKGROUND_POLL_INTERVAL)
                continue

            if response.status in {"failed", "cancelled", "incomplete"}:
                raise ValueError(response)

            if response.status == "completed":
                return response

            raise ValueError(response)

    @staticmethod
    def _get_tools(
        default_tools: list[dict[str, Any]] | None,
        user_tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        tools = (user_tools or [])[:]

        for tool in default_tools or []:
            if tool["type"] == "mcp":
                tool["server_url"] = tool["server_url"].format(**os.environ)

                try:
                    for k, v in tool["headers"].items():
                        tool["headers"][k] = v.format(**os.environ)

                except KeyError:
                    pass

                tools.append(tool)
                continue

            if tool["type"] in [x["type"] for x in tools]:
                continue

            tools.append(tool)

        return tools

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> ModelResponse:
        raise NotImplementedError()

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ModelResponse:
        raise NotImplementedError()

    def streaming(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> Iterator[GenericStreamingChunk]:
        raise NotImplementedError()

    async def astreaming(  # pylint: disable=too-many-branches,too-many-statements
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        outbound_aclient = AsyncOpenAI(api_key=api_key)

        background = optional_params.get("background", False)

        responses_params = {
            "model": optional_params.get("outbound_model", DEFAULT_OUTBOUND_MODEL),
            "input": self._get_input_from_messages(messages),
            "instructions": self._get_instructions_from_messages(messages),
        }

        if background:
            responses_params["background"] = True

        try:
            verbosity = optional_params["verbosity"]
        except KeyError:
            try:
                verbosity = optional_params["default_verbosity"]
            except KeyError:
                verbosity = None

        if verbosity:
            responses_params["verbosity"] = verbosity

        tools = self._get_tools(
            optional_params.get("default_tools", []), optional_params.get("tools", [])
        )

        if tools:
            responses_params["tools"] = tools

        try:
            responses_params["reasoning"] = {
                "effort": optional_params["reasoning_effort"]
            }
        except KeyError:
            pass

        prefix = optional_params.get("response_prefix", "").strip()

        if prefix:
            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": prefix + "\n",
                "tool_use": None,
                "usage": None,
            }

        conversation_id = self._get_conversation_id(messages)
        new_conv_id = conversation_id is None

        if not conversation_id:
            conversation_obj = await outbound_aclient.conversations.create()
            conversation_id = conversation_obj.id

        responses_params["conversation"] = conversation_id

        if new_conv_id:
            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": f"<conv_id={conversation_id}>\n",
                "tool_use": None,
                "usage": None,
            }

        if background:
            response_task = asyncio.create_task(
                self._background_responses(
                    outbound_aclient,
                    (await outbound_aclient.responses.create(**responses_params)).id,
                )
            )

        else:
            response_task = asyncio.create_task(
                outbound_aclient.responses.create(**responses_params)
            )

        has_heartbeat = False

        while True:
            sleep_task = asyncio.create_task(
                asyncio.sleep(
                    optional_params.get("heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)
                )
            )

            done, _ = await asyncio.wait(
                [response_task, sleep_task], return_when=asyncio.FIRST_COMPLETED
            )

            if response_task in done:
                break

            has_heartbeat = True

            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": optional_params.get("heartbeat_marker", DEFAULT_HEARTBEAT_MARKER),
                "tool_use": None,
                "usage": None,
            }

        if has_heartbeat:
            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": "\n",
                "tool_use": None,
                "usage": None,
            }

        output_text = self._extract_response_text(await response_task)

        for i in range(0, len(output_text), STREAMING_CHUNK_SIZE):
            chunk_text = output_text[i : i + STREAMING_CHUNK_SIZE]

            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": chunk_text,
                "tool_use": None,
                "usage": None,
            }

            await asyncio.sleep(STREAMING_CHUNK_DELAY)

        yield {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": "",
            "tool_use": None,
            "usage": {  # usage metrics placeholder
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }


class PerplexityBridge(CustomLLM):

    def _format_search_results(self, search_results: list[dict[str, Any]] | None) -> str:
        if not search_results:
            return ""

        formatted_results = ["\n## Sources:\n"]

        for i, result in enumerate(search_results, 1):
            title = result.get("title", "Unknown")
            url = result.get("url", "")
            date = result.get("date", "")

            formatted_results.append(f"{i}. **{title}**")

            if url:
                formatted_results.append(f"   URL: {url}")

            if date:
                formatted_results.append(f"   Date: {date}")

            formatted_results.append("")

        return "\n".join(formatted_results)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> ModelResponse:
        raise NotImplementedError()

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ModelResponse:
        raise NotImplementedError()

    def streaming(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> Iterator[GenericStreamingChunk]:
        raise NotImplementedError()

    async def astreaming(  # pylint: disable=too-many-branches
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        outbound_model = optional_params.pop("outbound_model")
        cut_think = optional_params.pop("cut_think", False)
        heartbeat_marker = optional_params.pop(
            "heartbeat_marker", DEFAULT_HEARTBEAT_MARKER
        )
        heartbeat_interval = optional_params.pop(
            "heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL
        )

        completion_params = {
            "model": f"perplexity/{outbound_model}",
            "messages": messages,
            "api_key": api_key,
            "stream": True,
            **optional_params,
        }

        stream_response = await litellm.acompletion(**completion_params)

        accumulated_search_results = None

        in_think = False
        think_is_cut = False
        has_heartbeat = False
        last_heartbeat_time = 0.0

        # dirty hack for basedpyright
        async for chunk in cast(AsyncIterator[Any], stream_response):
            if hasattr(chunk, "search_results") and chunk.search_results:
                accumulated_search_results = chunk.search_results

            if not chunk.choices or not chunk.choices[0].delta:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "") or ""
            finish_reason = chunk.choices[0].finish_reason

            is_finished = finish_reason is not None

            if not is_finished:
                if cut_think and not think_is_cut:
                    if "<think>" in content:
                        in_think = True
                        content = None

                    elif "</think>" in content:
                        in_think = False
                        think_is_cut = True

                        if has_heartbeat:
                            content = "\n"
                        else:
                            content = None

                    elif in_think:
                        current_time = time.time()

                        if current_time - last_heartbeat_time >= heartbeat_interval:
                            content = heartbeat_marker
                            has_heartbeat = True
                            last_heartbeat_time = current_time

                        else:
                            content = None

            else:
                if accumulated_search_results:
                    formatted_sources = self._format_search_results(
                        accumulated_search_results
                    )
                    content = content + formatted_sources

            if content is not None:
                yield {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "is_finished": is_finished,
                    "text": content,
                    "tool_use": None,
                    "usage": {  # usage metrics placeholder
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    },
                }


class AgentRouter(CustomLLM):

    def _history_model(self, messages: list[dict[str, Any]]) -> str | None:
        selected: str | None = None

        for message in messages:
            if message["role"] != "assistant":
                continue

            marker_text = str(message["content"])
            marker_index = marker_text.find("<model=")

            if marker_index == -1:
                continue

            end_index = marker_text.find(">", marker_index)

            if end_index == -1:
                continue

            selected = marker_text[marker_index + 7 : end_index]

        return selected

    def _first_user_content(self, messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message["role"] == "user":
                return str(message["content"])

        return ""

    async def _determine_model(self, messages: list[dict[str, Any]]) -> tuple[str, bool]:
        existing = self._history_model(messages)

        if existing in ROUTABLE_MODELS:
            return existing, False

        classify_response = await litellm.acompletion(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": self._first_user_content(messages)},
            ],
            temperature=0,
        )

        candidate_message = classify_response.choices[0].message.content

        if candidate_message is None:
            candidate_message = ""

        candidate_message = candidate_message.strip()

        for option in ROUTABLE_MODELS:
            if option in candidate_message:
                return option, True

        return ROUTABLE_MODELS[0], True

    def _call_params(
        self,
        selected_model: str,
        messages: list[dict[str, Any]],
        optional_params: dict[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "stream": stream,
        }

        for key in optional_params:
            params[key] = optional_params[key]

        params["stream"] = stream

        return params

    def _format_stream_chunk(
        self,
        chunk: Any,
        should_prefix: bool,
        selected_model: str,
        prefixed: bool,
    ) -> tuple[GenericStreamingChunk | None, bool]:
        if not chunk.choices:
            return None, prefixed

        choice = chunk.choices[0]
        delta = choice.delta
        updated_prefixed = prefixed

        if should_prefix and not prefixed:
            if delta.content is None:
                delta.content = f"<model={selected_model}>"
            else:
                delta.content = f"<model={selected_model}>" + delta.content

            updated_prefixed = True

        if delta.content is None:
            delta.content = ""

        text = delta.content
        finish_reason = choice.finish_reason

        if finish_reason is None:
            finish_reason = ""

        usage_value = None

        if hasattr(chunk, "usage"):
            usage_value = chunk.usage

        formatted_chunk: GenericStreamingChunk = {
            "finish_reason": finish_reason,
            "index": choice.index,
            "is_finished": bool(finish_reason),
            "text": text,
            "tool_use": None,
            "usage": usage_value,
        }

        return formatted_chunk, updated_prefixed

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
    ) -> ModelResponse:
        selected_model, should_prefix = await self._determine_model(messages)

        call_params = self._call_params(
            selected_model,
            messages,
            optional_params,
            False,
        )

        response = cast(ModelResponse, await litellm.acompletion(**call_params))

        if should_prefix:
            response_message = response.choices[0].message

            if response_message.content is None:
                response_message.content = ""

            response_message.content = (
                f"<model={selected_model}>" + response_message.content
            )

        return response

    async def astreaming(
        self,
        model: str,
        messages: list[dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict[str, Any],
        model_response: ModelResponse,
        print_verbose: Callable[..., None],
        encoding: Any,
        api_key: str | None,
        logging_obj: Any,
        optional_params: dict[str, Any],
        acompletion: Any = None,
        litellm_params: dict[str, Any] | None = None,
        logger_fn: Callable[..., Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        selected_model, should_prefix = await self._determine_model(messages)

        call_params = self._call_params(
            selected_model,
            messages,
            optional_params,
            True,
        )

        stream_response = await litellm.acompletion(**call_params)

        prefixed = False

        async for chunk in cast(AsyncIterator[Any], stream_response):
            formatted_chunk, prefixed = self._format_stream_chunk(
                chunk,
                should_prefix,
                selected_model,
                prefixed,
            )

            if formatted_chunk is None:
                continue

            yield formatted_chunk


openai_responses_bridge = OpenAIResponsesBridge()
perplexity_bridge = PerplexityBridge()
agent_router = AgentRouter()
