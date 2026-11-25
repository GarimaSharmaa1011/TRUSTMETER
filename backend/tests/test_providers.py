import asyncio

from app.config import ModelConfig, load_models
import app.providers as providers
from app.providers import parse_structured_answer
from httpx import Request, Response
from openai import RateLimitError


def test_json_confidence_is_parsed_only_when_valid():
    assert parse_structured_answer('{"final_answer":"42","confidence":87}') == ("42", 87.0)
    assert parse_structured_answer('{"final_answer":"42","confidence":"high"}') == ("42", None)
    assert parse_structured_answer('{"final_answer":"42","confidence":101}') == ("42", None)


def test_plain_text_answer_does_not_invent_confidence():
    assert parse_structured_answer("The answer is 42.") == ("The answer is 42.", None)


def test_gemini_is_optional_until_explicitly_included(monkeypatch):
    monkeypatch.setenv("TRUSTMETER_INCLUDE_OPTIONAL_MODELS", "")
    assert "gemini-flash" not in {model.id for model in load_models()}

    monkeypatch.setenv("TRUSTMETER_INCLUDE_OPTIONAL_MODELS", "gemini-flash")
    assert "gemini-flash" in {model.id for model in load_models()}


def test_provider_rate_limiter_shares_slots_by_provider(monkeypatch):
    model_one = ModelConfig(
        "one", "One", "https://provider.test/v1", "one", "SHARED_KEY",
        requests_per_minute=20,
    )
    model_two = ModelConfig(
        "two", "Two", "https://provider.test/v1/", "two", "SHARED_KEY",
        requests_per_minute=20,
    )
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    key = ("https://provider.test/v1", "SHARED_KEY")
    providers._next_provider_request.pop(key, None)
    monkeypatch.setattr(providers.asyncio, "sleep", record_sleep)

    async def reserve_two_slots():
        await providers._wait_for_provider_slot(model_one)
        await providers._wait_for_provider_slot(model_two)

    asyncio.run(reserve_two_slots())

    assert len(delays) == 1
    assert 2.9 <= delays[0] <= 3.0


def test_daily_quota_error_is_classified_from_provider_body():
    error = RateLimitError(
        "quota exceeded",
        response=Response(429, request=Request("POST", "https://provider.test")),
        body=[{
            "error": {
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                    }],
                }],
            },
        }],
    )

    assert providers._quota_window(error) == (
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    )
