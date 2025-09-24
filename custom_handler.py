#!/usr/bin/env python3

import os
import time
from collections.abc import Iterator, AsyncIterator, AsyncGenerator
from typing import Any, Callable, cast
from copy import deepcopy
import json
import re
import asyncio

from dotenv import load_dotenv

import httpx

from openai import AsyncOpenAI
from openai.types.responses import Response

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ChatCompletionUsageBlock, ModelResponse
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


ROUTER_SYSTEM_PROMPT = """\
# Role
You are an LLM Router in LibreChat.

# Task
Read the user's request and recommend one the most suitable model from the list \
of available models. Do not solve the task itself. Your audience: top-managers, \
management consultants, financial analysts, and data analysts.


# Available models
- gpt-5
- gpt-5-thinking
- gpt-5-thinking-high
- o4-mini-deep-research
- o3-deep-research
- perplexity-pro
- perplexity-pro-high
- perplexity-reasoning-pro-high
- perplexity-deep-research-high
- claude-sonnet
- claude-sonnet-thinking-high
- claude-opus-thinking-high

Notes:
- perplexity = sonar
- "-high" on perplexity means "web_search_options": {"search_context_size": "high"}
    - medium (w/o -high) -> default, best suited for general use cases
    - -high -> research, exploratory questions, or when citations and evidence coverage are critical


# Typical domains
- Management consulting
- Finance
- Analytics
- Market/competitive research
- General web search tasks
- Coding and technical analysis
- Complex reasoning tasks


# Instruction

Decide one best model:
- general queries -> gpt-5
- light web search, fast results -> perplexity-pro
- comprehensive web search -> perplexity-pro-high
- complex multi-step reasoning with web search -> gpt-5-thinking-high
- complex multi-step reasoning with deep analysis and coding -> claude-opus-thinking-high
- heavy long research with many sources -> perplexity-deep-research-high or o3-deep-research
- coding and technical tasks -> claude-sonnet
- large document processin -> claude-sonnet


# Output format

<think>
...your understanding of the task and rationale of decision...
</think>

<model>
...exactly one model from the list of available models...
</model>\
"""


def _g(o: Any, k: Any, d: Any | None = None) -> Any:
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


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
                match = re.search(r"<conv_id=(.*?)>", str(message["content"]))

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

    @classmethod
    def _get_tools(
        cls,
        default_tools: list[dict[str, Any]] | None,
        user_tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        tools = (cls._chat_completion_tools_to_responses(user_tools) or [])[:]

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

    @staticmethod
    def _chat_completion_tools_to_responses(
        tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """
        Chat Completions:
          {"type":"function","function":{"name","description"?, "parameters"?}}
        → Responses / GPT‑5 bridges (FunctionTool):
          {"type":"function","name","description"?, "parameters":{...}}
        """
        if not tools:
            return tools

        out_tools = []

        for t in tools or []:
            if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
                f = t["function"]
                name = f["name"]

                params = f.get("parameters") if isinstance(f.get("parameters"), dict) else {"type":"object","properties":{}}

                out_tools.append({
                    "type": "function",
                    "name": name,
                    "description": f.get("description",""),
                    "parameters": deepcopy(params)
                })

            else:
                out_tools.append(deepcopy(t))

        return out_tools

    @classmethod
    def _extract_from_responses(
        cls, resp: Any
    ) -> tuple[list[dict[str, Any]], str, ChatCompletionUsageBlock, str]:
        output = _g(resp, "output") or []
        tool_calls: list[dict[str, Any]] = []
        texts: list[str] = []

        for item in output:
            itype = _g(item, "type")

            # function/tool call items
            if itype in ("function_call", "tool_call", "tool_use"):
                name = _g(item, "name")

                if not name:
                    # FIXME: warning
                    continue

                call_id = _g(item, "call_id") or _g(item, "id") or f"call_{len(tool_calls)+1}"
                args = _g(item, "arguments")
                args_str = json.dumps(args, ensure_ascii=False, separators=(",", ":")) if isinstance(args, (dict, list)) else (args or "{}")
                tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": args_str}})

            # collect any text fragments
            ci = _g(item, "content")

            if isinstance(ci, str):
                texts.append(ci)

            elif isinstance(ci, list):
                for c in ci:
                    if isinstance(c, dict) and c.get("type") == "text":
                        texts.append(c.get("text") or "")

        # fallback to output_text
        if not texts:
            ot = _g(resp, "output_text")
            if isinstance(ot, str):
                texts = [ot]

        text = "".join(texts)
        usage = cls._normalize_usage(_g(resp, "usage"))

        if tool_calls:
            return tool_calls, "", usage, "tool_calls"

        return [], text, usage, "stop"

    async def _stream_chat_from_responses(
        self, resp: Any, chunk_size: int = STREAMING_CHUNK_SIZE
    ) -> AsyncGenerator[GenericStreamingChunk, None]:
        tool_calls, text, usage, finish_reason = self._extract_from_responses(resp)

        # If the model issued tool calls, emit them first and finish.
        if tool_calls:
            for tc in tool_calls:
                yield {
                    "finish_reason": "",
                    "index": 0,
                    "is_finished": False,
                    "text": "",
                    "tool_use": tc,   # <-- Chat Completions-style tool call
                    "usage": None,
                }

            yield {
                "finish_reason": finish_reason,  # "tool_calls"
                "index": 0,
                "is_finished": True,
                "text": "",
                "tool_use": None,
                "usage": usage,
            }

            return

        # Otherwise stream text chunks.
        if text:
            for i in range(0, len(text), chunk_size):
                yield {
                    "finish_reason": "",
                    "index": 0,
                    "is_finished": False,
                    "text": text[i:i+chunk_size],
                    "tool_use": None,
                    "usage": None,
                }

        # Final frame with usage.
        yield {
            "finish_reason": finish_reason,  # "stop" when only text
            "index": 0,
            "is_finished": True,
            "text": "",
            "tool_use": None,
            "usage": usage,
        }

    @staticmethod
    def _obj_to_dict(o: Any) -> dict[str, Any] | None:
        if o is None:
            return None

        if isinstance(o, dict):
            return o

        if hasattr(o, "model_dump"):
            return o.model_dump()

        if hasattr(o, "dict"):
            return o.dict()

        if hasattr(o, "__dict__"):
            return vars(o)

        return None

    @classmethod
    def _normalize_usage(cls, uobj) -> ChatCompletionUsageBlock:
        u = cls._obj_to_dict(uobj) or {}
        prompt = u.get("input_tokens") or u.get("prompt_tokens") or 0
        completion = u.get("output_tokens") or u.get("completion_tokens") or 0
        total = u.get("total_tokens") or (prompt or 0) + (completion or 0)

        return {
            "prompt_tokens": int(prompt or 0),
            "completion_tokens": int(completion or 0),
            "total_tokens": int(total or 0),
        }

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

        conversation_id = self._get_conversation_id(messages)
        new_conv_id = conversation_id is None

        if not conversation_id:
            conversation_obj = await outbound_aclient.conversations.create()
            conversation_id = conversation_obj.id

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

        responses_params["conversation"] = conversation_id

        if new_conv_id:
            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": f"`<conv_id={conversation_id}>`\n",
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

        async for ev in self._stream_chat_from_responses(await response_task):
            yield ev


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
                    "usage": None,
                }


class AgentRouter(CustomLLM):

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
        has_model_msg = False
        routed_to_model = None

        for message in messages:
            if message["role"] != "assistant":
                continue

            match = re.search(r"<model=(.*?)>", str(message["content"]))

            if match:
                has_model_msg = True
                routed_to_model = match.group(1)

        if routed_to_model is None:
            user_message = None

            for message in messages:
                if message["role"] == "user":
                    user_message = str(message["content"])

            if not user_message:
                raise ValueError(messages)

            response = await litellm.acompletion(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            s = str(response.choices[0].message.content)
            match = re.search(r"<model>(.*?)</model>", s, flags=re.DOTALL)

            if not match:
                raise ValueError(s)

            routed_to_model = "x-" + match.group(1).strip()

        if not has_model_msg:
            yield {
                "finish_reason": "",
                "index": 0,
                "is_finished": False,
                "text": f"`<model={routed_to_model}>`\n",
                "tool_use": None,
                "usage": None,
            }

        proxy_client = AsyncOpenAI(
            base_url=(f'http://{os.environ["SERVER_HOST"]}:'
                      f'{os.environ["SERVER_PORT"]}/v1'),
            api_key="dummy-api-key",
        )

        request_params = {
            "model": routed_to_model,
            "messages": messages,
            "stream": True,
        }

        if "max_tokens" in optional_params:
            request_params["max_tokens"] = optional_params["max_tokens"]

        if "temperature" in optional_params:
            request_params["temperature"] = optional_params["temperature"]

        if "search_context_size" in optional_params:
            request_params["search_context_size"] = optional_params["search_context_size"]

        stream_response = await proxy_client.chat.completions.create(**request_params)

        async for chunk in stream_response:
            if not chunk.choices or not chunk.choices[0].delta:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            content = getattr(delta, "content", "")
            finish_reason = getattr(choice, "finish_reason", "")

            yield {
                "finish_reason": finish_reason,
                "index": 0,
                "is_finished": finish_reason is not None,
                "text": content,
                "tool_use": None,
                "usage": getattr(chunk, "usage", None),
            }


openai_responses_bridge = OpenAIResponsesBridge()
perplexity_bridge = PerplexityBridge()
agent_router = AgentRouter()
