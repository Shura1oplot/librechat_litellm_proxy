#!/usr/bin/env python3

import os
import time
from collections.abc import Iterator, AsyncIterator
from typing import Any, Callable
import json
import re
import asyncio

from dotenv import load_dotenv

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import (GenericStreamingChunk,
                                 ChatCompletionToolCallChunk,
                                 ModelResponse)
from litellm.llms.custom_httpx.http_handler import (AsyncHTTPHandler,
                                                    HTTPHandler)

import httpx

from openai import AsyncOpenAI
from openai.types.responses import Response

import redis.asyncio as redis


_ = load_dotenv()


DEFAULT_OUTBOUND_MODEL = "gpt-5"
DEFAULT_HEARTBEAT_INTERVAL = 15.0
DEFAULT_HEARTBEAT_MARKER = "."

STREAMING_CHUNK_SIZE = 50
STREAMING_CHUNK_DELAY = 0.02

BACKGROUND_POLL_INTERVAL = 5.0

REDIS_DB_OPENAI = 0
REDIS_DB_ROUTER = 1

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
- Facts are time sensitive. Write the year near the link.
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


class OpenAIResponsesBridge(CustomLLM):

    @staticmethod
    async def _background_responses(aclient: AsyncOpenAI,
                                    _id: str) -> Response:
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

    def completion(self,
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
                   client: HTTPHandler | None = None) \
                   -> ModelResponse:
        raise NotImplementedError()

    async def acompletion(self,
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
                          client: AsyncHTTPHandler | None = None) \
                          -> ModelResponse:
        raise NotImplementedError()

    def streaming(self,
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
                  client: HTTPHandler | None = None) \
                  -> Iterator[GenericStreamingChunk]:
        raise NotImplementedError()

    async def astreaming(self,
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
                         client: AsyncHTTPHandler | None = None) \
                         -> AsyncIterator[GenericStreamingChunk]:
        input_: Any = None

        for message in reversed(messages):
            if message["role"] == "user":
                content = message["content"]

                if isinstance(content, str):
                    input_ = content
                    break

                if isinstance(content, list):
                    content_out = []

                    for content_item in content:
                        type_ = content_item["type"]

                        if type_ == "text":
                            content_out.append({
                                "type": "input_text",
                                "text": content_item["text"]})
                            continue

                        if type_ == "image_url":
                            content_out.append({
                                "type": "input_image",
                                "image_url": content_item["image_url"]["url"]})
                            continue

                        raise ValueError(content_item)

                    input_ = [{"role": "user",
                               "content": content_out}]

                    break

                raise ValueError(message)

            if message["role"] == "tool":
                t = message.get("content", "")

                if isinstance(t, list):
                    t = "".join(x.get("text","") for x in t
                                if isinstance(x, dict) and x.get("type") \
                                in ("text", "input_text", "output_text"))

                elif isinstance(t, dict) and "text" in t:
                    t = t["text"]
                else:
                    t = str(t)

                input_ = [{"type": "function_call_output",
                           "call_id": message["tool_call_id"],
                           "output": t}]

                break

        if not input_:
            raise ValueError(messages)

        instructions = DEFAULT_SYSTEM_PROMPT

        for message in messages:
            if message["role"] in ("system", "developer"):
                instructions = str(message["content"])
                break

        responses_params = {
            "model": optional_params.get("outbound_model", DEFAULT_OUTBOUND_MODEL),
            "input": input_,
            "instructions": instructions,
        }

        background = optional_params.get("background", False)

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

        tools = []

        for tool in optional_params.get("tools") or []:
            if isinstance(tool, dict) and tool.get("type") == "function" \
                    and isinstance(tool.get("function"), dict):
                func = tool["function"]
                name = func["name"]

                params = func.get("parameters") \
                    if isinstance(func.get("parameters"), dict) \
                    else {"type":"object","properties":{}}

                tools.append({"type": "function",
                              "name": name,
                              "description": func.get("description",""),
                              "parameters": params})

            else:
                tools.append(tool)

        for tool in optional_params.get("default_tools") or []:
            if tool["type"] == "mcp":
                tool["server_url"] = tool["server_url"].format(**os.environ)

                try:
                    for k, v in tool["headers"].items():
                        tool["headers"][k] = v.format(**os.environ)

                except KeyError:
                    pass

                tools.append(tool)

                continue

            if tool["type"] in (x["type"] for x in tools):
                continue

            if tool["type"] == "web_search":
                has_web_search = False

                for x in tools:
                    if x["type"] == "function" and x["name"] == "web_search":
                        has_web_search = True
                        break

                if not has_web_search:
                    tools.append(tool)

                continue

            tools.append(tool)

        if tools:
            responses_params["tools"] = tools

        try:
            responses_params["reasoning"] = {
                "effort": optional_params["reasoning_effort"]}
        except KeyError:
            pass

        headers = headers or {}

        librechat_conv_id = headers["x-librechat-conversation-id"]

        r = redis.Redis(host=os.environ["REDIS_HOST"],
                        port=int(os.environ["REDIS_PORT"]),
                        db=REDIS_DB_OPENAI,
                        decode_responses=True)

        conversation_id = None

        if (await r.exists(librechat_conv_id)):
            conversation_id = await r.get(librechat_conv_id)

        outbound_aclient = AsyncOpenAI(api_key=api_key)

        if not conversation_id:
            conversation_obj = await outbound_aclient.conversations.create()
            conversation_id = conversation_obj.id

            await r.set(librechat_conv_id, conversation_id)

            prefix = optional_params.get("response_prefix", "").strip()

            if prefix:
                yield {"finish_reason": "",
                       "index": 0,
                       "is_finished": False,
                       "text": prefix + "\n",
                       "tool_use": None,
                       "usage": None}

        await r.aclose()

        responses_params["conversation"] = conversation_id

        if background:
            response_task = asyncio.create_task(
                self._background_responses(
                    outbound_aclient,
                    (await outbound_aclient.responses.create(**responses_params)).id))

        else:
            response_task = asyncio.create_task(
                outbound_aclient.responses.create(**responses_params))

        has_heartbeat = False

        while True:
            sleep_task = asyncio.create_task(
                asyncio.sleep(
                    optional_params.get("heartbeat_interval",
                                        DEFAULT_HEARTBEAT_INTERVAL)))

            done, _ = await asyncio.wait(
                [response_task, sleep_task], return_when=asyncio.FIRST_COMPLETED)

            if response_task in done:
                break

            has_heartbeat = True

            yield {"finish_reason": "",
                   "index": 0,
                   "is_finished": False,
                   "text": optional_params.get("heartbeat_marker",
                                               DEFAULT_HEARTBEAT_MARKER),
                   "tool_use": None,
                   "usage": None}

        if has_heartbeat:
            yield {"finish_reason": "",
                   "index": 0,
                   "is_finished": False,
                   "text": "\n",
                   "tool_use": None,
                   "usage": None}

        response = await response_task

        tool_calls: list[ChatCompletionToolCallChunk] = []
        texts: list[str] = []

        for item in response.output or []:
            _type = item.type

            if _type == "function_call":
                args = item.arguments
                args_str = json.dumps(args,
                                      ensure_ascii=False,
                                      separators=(",", ":")) \
                    if isinstance(args, (dict, list)) else (args or "{}")

                tool_calls.append({"id": item.call_id,
                                   "type": "function",
                                   "function": {"name": item.name,
                                                "arguments": args_str},
                                   "index": len(tool_calls)})

            if not hasattr(item, "content"):
                continue

            if isinstance(item.content, str):
                texts.append(item.content)

            elif isinstance(item.content, list):
                for x in item.content:
                    if isinstance(x, dict) and x.get("type") == "text":
                        texts.append(x["text"])

                        if "annotations" in x:
                            texts.append("\n\nSources:\n")

                            for i, a in enumerate(x["annotations"], 1):
                                if a["type"] != "url_citation":
                                    continue

                                texts.append(f'{i}. {a["title"]}\n')
                                texts.append(f'   {a["url"]}\n\n')

        usage = response.usage or {}

        usage_prompt = usage.input_tokens or usage.prompt_tokens or 0
        usage_completion = usage.output_tokens or usage.completion_tokens or 0
        usage_total = usage.total_tokens or (usage_prompt + usage_completion)

        if tool_calls:
            for tc in tool_calls:
                yield {"finish_reason": "",
                       "index": 0,
                       "is_finished": False,
                       "text": "",
                       "tool_use": tc,
                       "usage": None}

            yield {"finish_reason": "tool_calls",
                   "index": 0,
                   "is_finished": True,
                   "text": "",
                   "tool_use": None,
                   "usage": {"prompt_tokens": int(usage_prompt),
                             "completion_tokens": int(usage_completion),
                             "total_tokens": int(usage_total)}}

            return

        if not texts:
            texts = [str(response.output_text)]

        for text in texts:
            for i in range(0, len(text), STREAMING_CHUNK_SIZE):
                yield {"finish_reason": "",
                       "index": 0,
                       "is_finished": False,
                       "text": text[i:i+STREAMING_CHUNK_SIZE],
                       "tool_use": None,
                       "usage": None}

        yield {"finish_reason": "stop",
               "index": 0,
               "is_finished": True,
               "text": "",
               "tool_use": None,
               "usage": {"prompt_tokens": int(usage_prompt),
                         "completion_tokens": int(usage_completion),
                         "total_tokens": int(usage_total)}}


class PerplexityBridge(CustomLLM):

    def completion(self,
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
                   client: HTTPHandler | None = None) \
                   -> ModelResponse:
        raise NotImplementedError()

    async def acompletion(self,
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
                          client: AsyncHTTPHandler | None = None) \
                          -> ModelResponse:
        raise NotImplementedError()

    def streaming(self,
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
                  client: HTTPHandler | None = None) \
                  -> Iterator[GenericStreamingChunk]:
        raise NotImplementedError()

    async def astreaming(self,
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
                         client: AsyncHTTPHandler | None = None) \
                         -> AsyncIterator[GenericStreamingChunk]:
        outbound_model = optional_params.pop("outbound_model")

        cut_think = optional_params.pop("cut_think", False)

        heartbeat_marker = optional_params.pop(
            "heartbeat_marker", DEFAULT_HEARTBEAT_MARKER)
        heartbeat_interval = optional_params.pop(
            "heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)

        completion_params = {"model": f"perplexity/{outbound_model}",
                             "messages": messages,
                             "api_key": api_key,
                             "stream": True,
                             **optional_params}

        stream_response = await litellm.acompletion(**completion_params)

        if isinstance(stream_response, ModelResponse):
            raise ValueError(stream_response)

        accumulated_search_results = None

        in_think = False
        think_is_cut = False
        has_heartbeat = False
        last_heartbeat_time = 0.0

        async for chunk in stream_response:
            if hasattr(chunk, "search_results") and chunk.search_results:
                accumulated_search_results = chunk.search_results

            if not chunk.choices or not chunk.choices[0].delta:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "") or ""
            finish_reason = chunk.choices[0].finish_reason or ""

            if not finish_reason:
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
                    formatted_sources = ["\n## Sources:\n"]

                    for i, result in enumerate(accumulated_search_results, 1):
                        title = result.get("title", "Unknown")
                        url = result.get("url", "")
                        date = result.get("date", "")

                        formatted_sources.append(f"{i}. **{title}**")

                        if url:
                            formatted_sources.append(f"   URL: {url}")

                        if date:
                            formatted_sources.append(f"   Date: {date}")

                        formatted_sources.append("")

                    content += "\n".join(formatted_sources)

            if content:
                yield {"finish_reason": finish_reason,
                       "index": 0,
                       "is_finished": bool(finish_reason),
                       "text": content,
                       "tool_use": None,
                       "usage": None}


class AgentRouter(CustomLLM):

    def completion(self,
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
                   client: HTTPHandler | None = None) \
                   -> ModelResponse:
        raise NotImplementedError()

    async def acompletion(self,
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
                          client: AsyncHTTPHandler | None = None) \
                          -> ModelResponse:
        raise NotImplementedError()

    def streaming(self,
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
                  client: HTTPHandler | None = None) \
                  -> Iterator[GenericStreamingChunk]:
        raise NotImplementedError()

    async def astreaming(self,
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
                         client: AsyncHTTPHandler | None = None) \
                         -> AsyncIterator[GenericStreamingChunk]:
        headers = headers or {}

        need_model_msg = False

        librechat_conv_id = headers["x-librechat-conversation-id"]

        r = redis.Redis(host=os.environ["REDIS_HOST"],
                        port=int(os.environ["REDIS_PORT"]),
                        db=REDIS_DB_ROUTER,
                        decode_responses=True)
        
        routed_to_model = None

        if (await r.exists(librechat_conv_id)):
            routed_to_model = await r.get(librechat_conv_id)

        if not routed_to_model:
            need_model_msg = True

            user_message = None

            for message in messages:
                if message["role"] == "user":
                    user_message = str(message["content"])

            if not user_message:
                raise ValueError(messages)

            response = await litellm.acompletion(
                model="gpt-5-mini",
                messages=[{"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                          {"role": "user", "content": user_message}])

            s = str(response.choices[0].message.content)
            match = re.search(r"<model>(.*?)</model>", s, flags=re.DOTALL)

            if not match:
                raise ValueError(s)

            routed_to_model = "x-" + match.group(1).strip()

            await r.set(librechat_conv_id, routed_to_model)

        if need_model_msg:
            yield {"finish_reason": "",
                   "index": 0,
                   "is_finished": False,
                   "text": f"_Routed to: {routed_to_model}_\n",
                   "tool_use": None,
                   "usage": None}

        proxy_client = AsyncOpenAI(
            base_url=(f'http://{os.environ["SERVER_HOST"]}:'
                      f'{os.environ["SERVER_PORT"]}/v1'),
            api_key="dummy-api-key")

        request_params = {"model": routed_to_model,
                          "messages": messages,
                          "stream": True}

        if "max_tokens" in optional_params:
            request_params["max_tokens"] = optional_params["max_tokens"]

        if "temperature" in optional_params:
            request_params["temperature"] = optional_params["temperature"]

        if "search_context_size" in optional_params:
            request_params["search_context_size"] = optional_params["search_context_size"]

        if "tools" in optional_params:
            if routed_to_model.startswith("x-perplexity"):
                yield {"finish_reason": "",
                       "index": 0,
                       "is_finished": False,
                       "text": ("_Perpexity does not support custom Librechat tools "
                                "(File Search, Interpreter), they are disabled for this dialog._\n"),
                       "tool_use": None,
                       "usage": None}
            
            else:
                request_params["tools"] = optional_params["tools"]

        if headers:
            request_params["extra_headers"] = headers
        
        stream_response = await proxy_client.chat.completions.create(**request_params)

        async for chunk in stream_response:
            if not chunk.choices or not chunk.choices[0].delta:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            content = getattr(delta, "content", "")
            finish_reason = getattr(choice, "finish_reason", "")

            yield {"finish_reason": finish_reason,
                   "index": 0,
                   "is_finished": finish_reason is not None,
                   "text": content,
                   "tool_use": None,
                   "usage": getattr(chunk, "usage", None)}


openai_responses_bridge = OpenAIResponsesBridge()
perplexity_bridge = PerplexityBridge()
agent_router = AgentRouter()
