import asyncio
import json

from app.config import ModelConfig
import train.evaluate as evaluation


def test_evaluation_trains_and_saves_from_mock_responses_only(tmp_path, monkeypatch):
    models = [
        ModelConfig(
            id=f"model-{index}",
            display_name=f"Model {index}",
            base_url="http://example.invalid/v1",
            model=f"model-{index}",
            api_key_env=f"TEST_KEY_{index}",
            requires_api_key=False,
        )
        for index in range(3)
    ]
    questions = [
        {
            "id": f"q-{index}",
            "question": f"Question {index}?",
            "choices": [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}],
            "answer_key": "A" if index % 2 == 0 else "B",
        }
        for index in range(120)
    ]
    monkeypatch.setattr(evaluation, "load_models", lambda: models)
    monkeypatch.setattr(evaluation, "load_arc_questions", lambda: questions)
    monkeypatch.setattr(evaluation, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(evaluation, "MODELS_DIR", tmp_path / "models")

    async def providers_reachable(configured_models):
        return [(model.display_name, True) for model in configured_models]

    async def fill_cache(sampled_questions, configured_models, cache_path, cache):
        for question in sampled_questions:
            responses = {}
            for index, model in enumerate(configured_models):
                if question["id"] == "q-0" and index == 2:
                    cache["failures"] = {
                        "q-0": {
                            model.id: {
                                "status": "failed",
                                "error": "Provider returned no valid ARC choice.",
                            }
                        }
                    }
                    continue
                answer = "A" if index < 2 else "B"
                confidence = (78 + index * 3) if (answer == question["answer_key"]) else (86 + index)
                responses[model.id] = {
                    "answer": answer,
                    "confidence": confidence,
                    "latency_ms": 10 + index,
                    "raw_response": json.dumps({"final_answer": answer, "confidence": confidence}),
                }
            cache["responses"][question["id"]] = responses
        evaluation._atomic_json(cache_path, cache)
        return cache

    monkeypatch.setattr(evaluation, "_check_providers", providers_reachable)
    monkeypatch.setattr(evaluation, "_collect_responses", fill_cache)

    metrics = evaluation.run_evaluation()

    assert metrics["evaluation"]["split_sizes"] == {
        "train": 60,
        "validation": 30,
        "test": 30,
    }
    assert metrics["evaluation"]["qualified_questions"] == 119
    assert sum(metrics["evaluation"]["qualified_split_sizes"].values()) == 119
    assert metrics["evaluation"]["recorded_failures"] == 1
    assert set(metrics["systems"]) == {
        "best_single_raw_confidence",
        "best_single_calibrated_confidence",
        "majority_vote",
        "trustmeter",
    }
    assert (tmp_path / "models" / "trust_estimator.joblib").exists()
    assert (tmp_path / "models" / "single_calibrators.joblib").exists()
    assert (tmp_path / "models" / "metrics.json").exists()


def test_cache_model_change_preserves_other_models_and_question_entries(tmp_path):
    cache_path = tmp_path / "responses.json"
    cache_path.write_text(json.dumps({
        "model_specs": {"groq-old": "openai/gpt-oss-20b", "gemini": "gemini-3-flash"},
        "responses": {
            "q1": {
                "groq-old": {"answer": "A"},
                "gemini": {"answer": "B"},
            },
            "q2": {"groq-old": {"answer": "C"}},
        },
    }), encoding="utf-8")
    models = [
        ModelConfig("groq-old", "Groq", "https://example.test/v1", "openai/gpt-oss-20b", "KEY"),
        ModelConfig("groq-new", "New Groq", "https://example.test/v1", "new-model", "KEY"),
    ]

    cache = evaluation._load_cache(cache_path, models)

    assert cache["responses"]["q1"]["groq-old"] == {"answer": "A"}
    assert cache["responses"]["q1"]["gemini"] == {"answer": "B"}
    assert cache["responses"]["q2"]["groq-old"] == {"answer": "C"}
    assert cache["model_specs"]["groq-new"] == "new-model"


def test_daily_quota_failure_is_recorded_and_blocks_repeat_calls(tmp_path, monkeypatch):
    models = [
        ModelConfig("groq", "Groq", "http://example.test/v1", "groq-model", "KEY"),
        ModelConfig("gemini", "Gemini", "http://example.test/v1", "gemini-model", "KEY"),
    ]
    questions = [
        {
            "id": f"q-{index}",
            "question": f"Question {index}?",
            "choices": [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}],
        }
        for index in range(2)
    ]
    cache_path = tmp_path / "responses.json"
    cache = {"model_specs": evaluation._cache_specs(models), "responses": {}, "failures": {}}
    calls = {"groq": 0, "gemini": 0}

    async def fake_call_model(model, prompt, *, choice_prompt=None):
        calls[model.id] += 1
        if model.id == "gemini":
            return {
                "model_id": model.id,
                "answer": None,
                "error": "Provider request failed after retries (daily quota exceeded (GenerateRequestsPerDay)).",
            }
        return {
            "model_id": model.id,
            "answer": "A",
            "confidence": 90,
            "latency_ms": 1,
            "raw_response": '{"final_answer":"A","confidence":90}',
        }

    monkeypatch.setattr(evaluation, "call_model", fake_call_model)
    asyncio.run(evaluation._collect_responses(questions, models, cache_path, cache))

    assert calls == {"groq": 2, "gemini": 1}
    assert cache["failures"]["q-0"]["gemini"]["status"] == "failed"
    assert cache["failures"]["q-1"]["gemini"]["status"] == "not_attempted_daily_quota"


def test_invalid_cached_arc_choice_does_not_qualify_question():
    questions = [{
        "id": "q1",
        "question": "Question?",
        "choices": [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}],
        "answer_key": "A",
    }]
    cache = {
        "responses": {
            "q1": {
                "one": {"answer": "A"},
                "two": {"answer": "I think it is A"},
                "three": {"answer": "A"},
            },
        },
        "failures": {},
    }

    split_data = evaluation._prepare_samples(
        questions, {"train": ["q1"], "validation": [], "test": []},
        cache, ["one", "two", "three"],
    )

    assert split_data["train"] == []
    assert cache["failures"]["q1"]["two"]["status"] == "failed"


def test_nonempty_cached_response_is_not_called_again(tmp_path, monkeypatch):
    models = [
        ModelConfig("one", "One", "http://example.test/v1", "one", "KEY"),
    ]
    questions = [{
        "id": "q1",
        "question": "Question?",
        "choices": [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}],
    }]
    cache = {
        "model_specs": evaluation._cache_specs(models),
        "responses": {"q1": {"one": {"answer": "cached raw answer"}}},
        "failures": {},
    }

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("A cached model response must not be called again.")

    monkeypatch.setattr(evaluation, "call_model", fail_if_called)
    asyncio.run(evaluation._collect_responses(
        questions, models, tmp_path / "responses.json", cache
    ))

    assert cache["responses"]["q1"]["one"]["answer"] == "cached raw answer"


def test_cache_only_evaluation_trains_from_two_models_without_provider_calls(tmp_path, monkeypatch):
    models = [
        ModelConfig("groq-one", "Groq One", "https://example.test/v1", "model-one", "KEY", requires_api_key=False),
        ModelConfig("groq-two", "Groq Two", "https://example.test/v1", "model-two", "KEY", requires_api_key=False),
    ]
    questions = [
        {
            "id": f"q-{index}",
            "question": f"Question {index}?",
            "choices": [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}],
            "answer_key": "A" if index % 2 == 0 else "B",
        }
        for index in range(120)
    ]
    monkeypatch.setattr(evaluation, "load_models", lambda: models)
    monkeypatch.setattr(evaluation, "load_arc_questions", lambda: questions)
    monkeypatch.setattr(evaluation, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(evaluation, "MODELS_DIR", tmp_path / "models")
    cache_path = tmp_path / "data" / "responses_cache.json"
    cache_path.parent.mkdir()
    cache_path.write_text(json.dumps({
        "model_specs": evaluation._cache_specs(models),
        "responses": {
            question["id"]: {
                "groq-one": {"answer": "A", "confidence": 80, "latency_ms": 1},
                "groq-two": {"answer": "B", "confidence": 80, "latency_ms": 1},
            }
            for question in questions
        },
    }), encoding="utf-8")

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("Cache-only mode must not contact a provider.")

    monkeypatch.setattr(evaluation, "_check_providers", fail_if_called)
    monkeypatch.setattr(evaluation, "_collect_responses", fail_if_called)
    metrics = evaluation.run_evaluation(cache_only=True)

    assert metrics["evaluation"]["response_source"] == "cache_only"
    assert metrics["evaluation"]["qualified_questions"] == 120
    assert metrics["evaluation"]["qualified_split_sizes"] == {
        "train": 60,
        "validation": 30,
        "test": 30,
    }
