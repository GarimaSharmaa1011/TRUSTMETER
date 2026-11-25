"""OpenAI-compatible provider calls with bounded retries and safe parsing."""

import asyncio
import json
import os
import random
import re
import threading
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.config import ModelConfig

CALL_TIMEOUT_SECONDS = max(1, int(os.getenv("PROVIDER_TIMEOUT_SECONDS", "30")))
MAX_ATTEMPTS = max(1, int(os.getenv("PROVIDER_MAX_ATTEMPTS", "3")))
RETRY_BASE_SECONDS = max(0.1, float(os.getenv("PROVIDER_RETRY_BASE_SECONDS", "1")))
RETRY_MAX_SECONDS = max(
    RETRY_BASE_SECONDS, float(os.getenv("PROVIDER_RETRY_MAX_SECONDS", "30"))
)
_rate_limit_lock = threading.Lock()
_next_provider_request: dict[tuple[str, str], float] = {}


async def _wait_for_provider_slot(model: ModelConfig) -> None:
    key = (model.base_url.rstrip("/"), model.api_key_env)
    interval = 60.0 / model.requests_per_minute
    with _rate_limit_lock:
        now = time.monotonic()
        next_request = max(now, _next_provider_request.get(key, now))
        delay = next_request - now
        _next_provider_request[key] = next_request + interval
    if delay:
        await asyncio.sleep(delay)


def _quota_window(error: RateLimitError) -> str | None:
    def find_quota_id(value: Any) -> str | None:
        if isinstance(value, dict):
            quota_id = value.get("quotaId")
            if isinstance(quota_id, str):
                return quota_id
            for item in value.values():
                found = find_quota_id(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = find_quota_id(item)
                if found:
                    return found
        return None

    return find_quota_id(getattr(error, "body", None))


def _retry_after_seconds(error: RateLimitError) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("retry-after") if headers else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_structured_answer(content: str) -> tuple[str | None, float | None]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return (text or None), None
    if not isinstance(parsed, dict):
        return None, None
    answer = parsed.get("final_answer", parsed.get("answer"))
    if answer is not None and not isinstance(answer, str):
        answer = str(answer)
    confidence = parsed.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = None
    elif 0 <= confidence <= 100:
        confidence = float(confidence)
    else:
        confidence = None
    return (answer.strip() if answer and answer.strip() else None), confidence


def _content_from_response(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return ""


async def call_model(
    model: ModelConfig,
    question: str,
    *,
    choice_prompt: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not model.api_key:
        return {
            "model_id": model.id,
            "answer": None,
            "confidence": None,
            "latency_ms": None,
            "error": f"API key is not configured (set {model.api_key_env} in backend/.env).",
        }

    system = (
        "Answer the question. Return only a JSON object with exactly these fields: "
        '{"final_answer":"a concise answer","confidence":0}. '
        "Confidence must be your self-reported probability from 0 to 100 that your answer is correct. "
        "Do not include explanatory text outside the JSON."
    )
    if choice_prompt:
        system += " " + choice_prompt

    client = AsyncOpenAI(
        api_key=model.api_key,
        base_url=model.base_url,
        timeout=CALL_TIMEOUT_SECONDS,
        max_retries=0,
    )
    last_error = "Provider request failed."
    try:
        for attempt in range(MAX_ATTEMPTS):
            try:
                await _wait_for_provider_slot(model)
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": question},
                        ],
                        temperature=0,
                    ),
                    timeout=CALL_TIMEOUT_SECONDS + 1,
                )
                raw = _content_from_response(response)
                answer, confidence = parse_structured_answer(raw)
                return {
                    "model_id": model.id,
                    "answer": answer,
                    "confidence": confidence,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": None if answer else "Provider returned no parseable answer.",
                    "raw_response": raw,
                }
            except RateLimitError as exc:
                quota_id = _quota_window(exc)
                if quota_id and "perday" in quota_id.lower():
                    last_error = f"daily quota exceeded ({quota_id})"
                    break
                last_error = "RateLimitError"
                if attempt + 1 < MAX_ATTEMPTS:
                    exponential_delay = min(
                        RETRY_BASE_SECONDS * (2**attempt), RETRY_MAX_SECONDS
                    )
                    retry_after = _retry_after_seconds(exc)
                    if retry_after is not None and retry_after > RETRY_MAX_SECONDS:
                        last_error = f"RateLimitError (retry-after {retry_after:g}s exceeds retry cap)"
                        break
                    delay = max(exponential_delay, retry_after or 0.0)
                    await asyncio.sleep(delay)
            except (APITimeoutError, APIConnectionError, asyncio.TimeoutError) as exc:
                last_error = type(exc).__name__
                if attempt + 1 < MAX_ATTEMPTS:
                    await asyncio.sleep(min(2**attempt + random.random() * 0.25, 4))
            except Exception as exc:
                last_error = type(exc).__name__
                break
    finally:
        await client.close()
    return {
        "model_id": model.id,
        "answer": None,
        "confidence": None,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "error": f"Provider request failed after retries ({last_error}).",
    }


async def ask_models(
    models: list[ModelConfig],
    question: str,
    *,
    choice_prompt: str | None = None,
) -> list[dict[str, Any]]:
    tasks = [
        call_model(model, question, choice_prompt=choice_prompt)
        for model in models
    ]
    return list(await asyncio.gather(*tasks))
