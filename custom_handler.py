#!/usr/bin/env python3

import asyncio
import concurrent.futures
import queue
import threading
import time
from typing import Any, Callable, cast
from collections.abc import Iterator, AsyncIterator
import re

import httpx

from openai import OpenAI, AsyncOpenAI

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


DEFAULT_OUTBOUND_MODEL = "gpt-5"
DEFAULT_HEARTBEAT_INTERVAL = 15.0
DEFAULT_HEARTBEAT_MARKER = "."

STREAMING_CHUNK_SIZE = 50
STREAMING_CHUNK_DELAY = 0.02


class OpenAIResponsesBridge(CustomLLM):

    def __init__(self) -> None:
        super().__init__()

        self.openai_client = OpenAI()
        self.async_openai_client = AsyncOpenAI()

    def _get_input_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content_items = self._build_message_content(message)

            if not content_items:
                continue

            return [{"role": "user", "content": content_items}]

        raise ValueError("No user message with content found in messages")

    def _build_message_content(self, message: dict[str, Any]) -> list[Any]:
        content_items = self._normalize_content_field(message.get("content"))

        attachments = self._collect_attachment_items(message)

        if attachments:
            content_items.extend(attachments)

        return content_items

    def _normalize_content_field(self, content: Any) -> list[Any]:
        if content is None:
            return []

        if isinstance(content, str):
            return self._build_text_content({"text": content})

        if isinstance(content, list):
            normalized: list[Any] = []

            for item in content:
                normalized.extend(self._normalize_content_item(item))

            return normalized

        return self._build_text_content({"text": str(content)})

    def _normalize_content_item(self, item: Any) -> list[Any]:
        if item is None:
            return []

        if isinstance(item, str):
            return self._build_text_content({"text": item})

        if not isinstance(item, dict):
            return self._build_text_content({"text": str(item)})

        item_type = str(item.get("type") or "")

        selected_builder: Callable[[dict[str, Any]], list[Any]] | None

        if item_type in ("text", "input_text", ""):
            selected_builder = self._build_text_content
        else:
            builders: dict[str, Callable[[dict[str, Any]], list[Any]]] = {
                "input_image": self._build_image_content,
                "image_url": self._build_image_content,
                "image": self._build_image_content,
                "input_file": self._build_file_content,
                "file": self._build_file_content,
                "document": self._build_file_content,
                "input_audio": self._build_audio_content,
                "audio": self._build_audio_content,
            }

            selected_builder = builders.get(item_type)

            if selected_builder is None:
                if "image_url" in item:
                    selected_builder = self._build_image_content
                elif any(key in item for key in ("file_id", "file_url", "file_data")):
                    selected_builder = self._build_file_content
                elif "text" in item:
                    selected_builder = self._build_text_content
                else:
                    selected_builder = None

        if selected_builder is None:
            return self._build_text_content({"text": str(item)})

        return selected_builder(item)

    def _build_text_content(self, item: dict[str, Any]) -> list[Any]:
        text = item.get("text")

        if text is None:
            text = item.get("value") or item.get("content")

        if text is None:
            return []

        return [{"type": "input_text", "text": str(text)}]

    def _build_image_content(self, item: dict[str, Any]) -> list[Any]:
        detail = self._extract_image_detail(item)
        file_id = self._first_non_empty(item, ("file_id", "id"))
        image_url, detail_from_url = self._extract_image_url(item.get("image_url"))

        if not detail and detail_from_url:
            detail = detail_from_url

        if not image_url:
            direct_url = item.get("url")

            if isinstance(direct_url, str):
                image_url = direct_url

        if not file_id and not image_url:
            base64_data = item.get("image_base64") or item.get("b64_json")

            if base64_data:
                mime_type = item.get("mime_type") or item.get("media_type") or "image/png"
                image_url = f"data:{mime_type};base64,{base64_data}"

        image_item: dict[str, Any] = {"type": "input_image"}

        if detail in {"low", "high", "auto"}:
            image_item["detail"] = detail

        if file_id:
            image_item["file_id"] = str(file_id)

        if image_url:
            image_item["image_url"] = str(image_url)

        return [image_item]

    def _extract_image_detail(self, item: dict[str, Any]) -> str | None:
        detail = item.get("detail")

        if detail in {"low", "high", "auto"}:
            return cast(str, detail)

        return None

    def _extract_image_url(self, value: Any) -> tuple[str | None, str | None]:
        if isinstance(value, dict):
            url = value.get("url") or value.get("image_url") or value.get("href")
            detail = value.get("detail")

            return (
                str(url) if isinstance(url, str) else None,
                detail if isinstance(detail, str) else None,
            )

        if isinstance(value, str):
            return (value, None)

        return (None, None)

    def _build_file_content(self, item: dict[str, Any]) -> list[Any]:
        file_item: dict[str, Any] = {"type": "input_file"}

        file_id = self._first_non_empty(item, ("file_id", "id"))

        if file_id:
            file_item["file_id"] = str(file_id)

        file_url = item.get("file_url") or item.get("url")

        if isinstance(file_url, str):
            file_item["file_url"] = file_url

        file_data = item.get("file_data")

        if isinstance(file_data, str):
            file_item["file_data"] = file_data

        filename = item.get("filename")

        if isinstance(filename, str):
            file_item["filename"] = filename

        return [file_item]

    def _build_audio_content(self, item: dict[str, Any]) -> list[Any]:
        audio_payload = item.get("input_audio")

        if not isinstance(audio_payload, dict):
            audio_payload = {
                "data": item.get("data"),
                "format": item.get("format"),
            }

        data = audio_payload.get("data")

        if not isinstance(data, str) or not data:
            return []

        format_hint = audio_payload.get("format")

        if not isinstance(format_hint, str) or not format_hint:
            format_hint = "wav"

        return [
            {
                "type": "input_audio",
                "input_audio": {"data": data, "format": format_hint},
            }
        ]

    def _collect_attachment_items(self, message: dict[str, Any]) -> list[Any]:
        attachment_items: list[Any] = []

        for key in ("attachments", "files", "documents"):
            entries = message.get(key)

            if isinstance(entries, list):
                for attachment in entries:
                    attachment_items.extend(self._file_items_from_any(attachment))
            elif entries is not None:
                attachment_items.extend(self._file_items_from_any(entries))

        attachment_items.extend(self._file_items_from_any(message.get("file_ids")))
        attachment_items.extend(self._file_items_from_any(message.get("file")))

        return attachment_items

    def _file_items_from_any(self, value: Any) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, str):
            return [{"type": "input_file", "file_id": value}]

        if isinstance(value, list):
            file_items: list[Any] = []

            for entry in value:
                file_items.extend(self._file_items_from_any(entry))

            return file_items

        if isinstance(value, dict):
            return self._build_file_content(value)

        return []

    def _first_non_empty(self, item: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            value = item.get(key)

            if value:
                return value

        return None

    def _get_instructions_from_messages(
        self, messages: list[dict[str, Any]]
    ) -> str | None:
        for message in messages:
            if message.get("role") == "system":
                return str(message["content"])

        return None

    def _heartbeat(
        self,
        stop_event: threading.Event,
        heartbeat_queue: queue.Queue,
        heartbeat_event: threading.Event,
        heartbeat_interval: float,
        heartbeat_marker: str,
    ) -> None:
        while not stop_event.wait(heartbeat_interval):
            if not stop_event.is_set():
                heartbeat_queue.put(heartbeat_marker)
                heartbeat_event.set()

    def _extract_response_text(self, response: Any) -> str:
        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                if hasattr(output_item, "content") and output_item.content:
                    for content_item in output_item.content:
                        if hasattr(content_item, "text") and content_item.text:
                            return str(content_item.text)

        elif hasattr(response, "output_text"):
            return response.output_text

        return str(response)

    def _convert_response_to_completion(
        self,
        response: Any,
        model: str,
    ) -> dict[str, Any]:
        completion_response = {
            "id": f"chatcmpl-{response.id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self._extract_response_text(response),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return completion_response

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
        responses_params = {
            "model": optional_params.get("outbound_model", DEFAULT_OUTBOUND_MODEL),
            "instructions": self._get_instructions_from_messages(messages),
            "input": self._get_input_from_messages(messages),
        }

        if optional_params.get("tools"):
            responses_params["tools"] = optional_params["tools"]

        if optional_params.get("reasoning_effort"):
            responses_params["reasoning"] = {
                "effort": optional_params["reasoning_effort"]
            }

        response = self.openai_client.responses.create(**responses_params)

        completion_dict = self._convert_response_to_completion(response, model)

        prefix = optional_params.get("response_prefix", "")

        if prefix:
            prefix = prefix + "\n"

        content = completion_dict["choices"][0]["message"]["content"]

        model_response.choices = [
            litellm.Choices(
                finish_reason=completion_dict["choices"][0]["finish_reason"],
                index=completion_dict["choices"][0]["index"],
                message=litellm.Message(
                    role=completion_dict["choices"][0]["message"]["role"],
                    content=prefix + content,
                ),
            )
        ]

        model_response.model = completion_dict["model"]

        model_response.usage = litellm.Usage(
            completion_tokens=completion_dict["usage"]["completion_tokens"],
            prompt_tokens=completion_dict["usage"]["prompt_tokens"],
            total_tokens=completion_dict["usage"]["total_tokens"],
        )

        return model_response

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
        responses_params = {
            "model": optional_params.get("outbound_model", DEFAULT_OUTBOUND_MODEL),
            "instructions": self._get_instructions_from_messages(messages),
            "input": self._get_input_from_messages(messages),
        }

        if optional_params.get("tools"):
            responses_params["tools"] = optional_params["tools"]

        if optional_params.get("reasoning_effort"):
            responses_params["reasoning"] = {
                "effort": optional_params["reasoning_effort"]
            }

        response = await self.async_openai_client.responses.create(**responses_params)

        completion_dict = self._convert_response_to_completion(response, model)

        prefix = optional_params.get("response_prefix", "")

        if prefix:
            prefix = prefix + "\n"

        content = completion_dict["choices"][0]["message"]["content"]

        model_response.choices = [
            litellm.Choices(
                finish_reason=completion_dict["choices"][0]["finish_reason"],
                index=completion_dict["choices"][0]["index"],
                message=litellm.Message(
                    role=completion_dict["choices"][0]["message"]["role"],
                    content=prefix + content,
                ),
            )
        ]

        model_response.model = completion_dict["model"]

        model_response.usage = litellm.Usage(
            completion_tokens=completion_dict["usage"]["completion_tokens"],
            prompt_tokens=completion_dict["usage"]["prompt_tokens"],
            total_tokens=completion_dict["usage"]["total_tokens"],
        )

        return model_response

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
        responses_params = {
            "model": optional_params.get("outbound_model", DEFAULT_OUTBOUND_MODEL),
            "instructions": self._get_instructions_from_messages(messages),
            "input": self._get_input_from_messages(messages),
        }

        if optional_params.get("tools"):
            responses_params["tools"] = optional_params["tools"]

        if optional_params.get("reasoning_effort"):
            responses_params["reasoning"] = {
                "effort": optional_params["reasoning_effort"]
            }

        prefix = optional_params.get("response_prefix", "")

        if prefix:
            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": prefix + "\n",
                "tool_use": None,
                "usage": None,
            }

        stop_event = threading.Event()
        heartbeat_queue = queue.Queue()
        heartbeat_event = threading.Event()

        heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            args=[
                stop_event,
                heartbeat_queue,
                heartbeat_event,
                optional_params.get("heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL),
                optional_params.get("heartbeat_marker", DEFAULT_HEARTBEAT_MARKER),
            ],
        )
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self.openai_client.responses.create, **responses_params
            )

            while not future.done():
                try:
                    yield {
                        "finish_reason": "",
                        "index": 0,
                        "is_finished": False,
                        "text": heartbeat_queue.get(timeout=0.1),
                        "tool_use": None,
                        "usage": None,
                    }

                except queue.Empty:
                    pass

            response = future.result()

        stop_event.set()

        if heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=1.0)

        output_text = self._extract_response_text(response)

        if heartbeat_event.is_set():
            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": "\n",
                "tool_use": None,
                "usage": None,
            }

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

            time.sleep(STREAMING_CHUNK_DELAY)

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
        responses_params = {
            "model": optional_params.get("outbound_model", DEFAULT_OUTBOUND_MODEL),
            "instructions": self._get_instructions_from_messages(messages),
            "input": self._get_input_from_messages(messages),
        }

        if optional_params.get("tools"):
            responses_params["tools"] = optional_params["tools"]

        if optional_params.get("reasoning_effort"):
            responses_params["reasoning"] = {
                "effort": optional_params["reasoning_effort"]
            }

        prefix = optional_params.get("response_prefix", "")

        if prefix:
            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": prefix + "\n",
                "tool_use": None,
                "usage": None,
            }

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
        outbound_model = optional_params.pop("outbound_model")
        cut_think = optional_params.pop("cut_think", False)

        completion_params = {
            "model": f"perplexity/{outbound_model}",
            "messages": messages,
            "api_key": api_key,
            **optional_params,
        }

        response = litellm.completion(**completion_params)

        content = response.choices[0].message.content or ""

        if cut_think:
            content = re.sub(
                r"^<think>.*?</think>", "", content, count=1, flags=re.DOTALL
            )

        search_results = getattr(response, "search_results", None)

        if search_results:
            formatted_sources = self._format_search_results(search_results)
            content = content + formatted_sources

        model_response.choices = [
            litellm.Choices(
                finish_reason=response.choices[0].finish_reason,
                index=response.choices[0].index,
                message=litellm.Message(
                    role="assistant",
                    content=content,
                ),
            )
        ]

        model_response.model = response.model
        model_response.usage = response.usage

        return model_response

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
        outbound_model = optional_params.pop("outbound_model")
        cut_think = optional_params.pop("cut_think", False)

        completion_params = {
            "model": f"perplexity/{outbound_model}",
            "messages": messages,
            "api_key": api_key,
            **optional_params,
        }

        response = await litellm.acompletion(**completion_params)

        content = response.choices[0].message.content or ""

        if cut_think:
            content = re.sub(
                r"^<think>.*?</think>", "", content, count=1, flags=re.DOTALL
            )

        search_results = getattr(response, "search_results", None)

        if search_results:
            formatted_sources = self._format_search_results(search_results)
            content = content + formatted_sources

        model_response.choices = [
            litellm.Choices(
                finish_reason=response.choices[0].finish_reason,
                index=response.choices[0].index,
                message=litellm.Message(
                    role="assistant",
                    content=content,
                ),
            )
        ]

        model_response.model = response.model
        model_response.usage = response.usage

        return model_response

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

        stream_response = litellm.completion(**completion_params)

        accumulated_search_results = None

        in_think = False
        think_is_cut = False
        has_heartbeat = False
        last_heartbeat_time = 0.0

        for chunk in stream_response:
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
                    "finish_reason": finish_reason or "",
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
