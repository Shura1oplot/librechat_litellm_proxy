#!/usr/bin/env python3

import asyncio
import time
from typing import Any, Callable, cast
from collections.abc import Iterator, AsyncIterator
import re

import httpx

from openai import AsyncOpenAI, Response

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


DEFAULT_OUTBOUND_MODEL = "gpt-5"
DEFAULT_HEARTBEAT_INTERVAL = 15.0
DEFAULT_HEARTBEAT_MARKER = "."

STREAMING_CHUNK_SIZE = 50
STREAMING_CHUNK_DELAY = 0.02

BACKGROUND_POLL_INTERVAL = 5.0


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
            if message["role"] == "user":
                match = re.search(r"<conv_id=(.*)>", str(message["content"]))

                if match:
                    return match.group(1)

        return None

    @staticmethod
    def _get_instructions_from_messages(
        messages: list[dict[str, Any]]
    ) -> str | None:
        for message in messages:
            if message.get("role") == "system":
                return str(message["content"])

        return None

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
        outbound_aclient = AsyncOpenAI(api_key=api_key)

        background = optional_params.get("background", False)

        responses_params = {
            "model": optional_params.get("outbound_model", DEFAULT_OUTBOUND_MODEL),
            "instructions": self._get_instructions_from_messages(messages),
            "input": self._get_input_from_messages(messages),
        }

        if background:
            responses_params["background"] = True

        try:
            responses_params["tools"] = optional_params["tools"]
        except KeyError:
            pass

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
            conversation_id = await outbound_aclient.conversations.create()

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
            response_task = asyncio.create_task(self._background_responses(
                outbound_aclient,
                await outbound_aclient.responses.create(**responses_params)
            ))

        else:
            response_task = asyncio.create_task(
                self.async_openai_client.responses.create(**responses_params)
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
            "usage": {  # TODO
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
                    "usage": {  # TODO
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    },
                }


openai_responses_bridge = OpenAIResponsesBridge()
perplexity_bridge = PerplexityBridge()
