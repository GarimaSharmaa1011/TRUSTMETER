"""Download ARC-Challenge, cache model responses, and train/evaluate TrustMeter."""

import asyncio
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
import httpx
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import load_models  # noqa: E402
from app.providers import call_model  # noqa: E402
from app.trust import calculate_trust_features, group_answers, trust_feature_names  # noqa: E402
from utils.calibration import SigmoidCalibratedClassifier  # noqa: E402
from utils.metrics import compute_ece  # noqa: E402

DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
SEED = int(os.getenv("ARC_RANDOM_SEED", "42"))
SAMPLE_SIZE = max(1, int(os.getenv("ARC_SAMPLE_SIZE", "300")))
DEFAULT_THRESHOLD = float(os.getenv("TRUST_THRESHOLD", "0.70"))


def split_question_ids(question_ids: list[str], seed: int = SEED) -> dict[str, list[str]]:
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Question IDs must be unique before splitting.")
    shuffled = np.random.default_rng(seed).permutation(question_ids).tolist()
    train_end = int(len(shuffled) * 0.50)
    validation_end = train_end + int(len(shuffled) * 0.25)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def load_arc_questions(sample_size: int = SAMPLE_SIZE, seed: int = SEED) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    selected = dataset.shuffle(seed=seed).select(range(min(sample_size, len(dataset))))
    questions = []
    for row in selected:
        choices = row["choices"]
        options = list(zip(choices["label"], choices["text"]))
        questions.append({
            "id": str(row["id"]),
            "question": str(row["question"]),
            "choices": [{"label": str(label), "text": str(text)} for label, text in options],
            "answer_key": str(row["answerKey"]),
        })
    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("ARC-Challenge sample contains duplicate question IDs.")
    return questions


def format_arc_prompt(question: dict[str, Any]) -> tuple[str, str]:
    options = "\n".join(f"{item['label']}. {item['text']}" for item in question["choices"])
    prompt = f"Question: {question['question']}\nChoices:\n{options}\nSelect the correct choice."
    instruction = (
        "This is a multiple-choice evaluation. Put only the exact choice label "
        "(for example A, B, C, or D) in final_answer."
    )
    return prompt, instruction


def extract_choice(answer: str | None, choices: list[dict[str, str]]) -> str | None:
    if not answer:
        return None
    labels = [choice["label"].strip().upper() for choice in choices]
    text = answer.strip().upper()
    exact = re.fullmatch(r"[\s\(\[]*([A-Z0-9]+)[\s\)\].:]*", text)
    if exact and exact.group(1) in labels:
        return exact.group(1)
    explicit = re.search(r"\b(?:ANSWER|CHOICE)\s*(?:IS|:)?\s*([A-Z0-9]+)\b", text)
    if explicit and explicit.group(1) in labels:
        return explicit.group(1)
    return None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _cache_specs(models) -> dict[str, str]:
    return {model.id: model.model for model in models}


async def _check_provider(model):
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                model.base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {model.api_key}"},
            )
        return model.display_name, response.status_code < 400 or response.status_code == 429
    except httpx.HTTPError:
        return model.display_name, False


async def _check_providers(models):
    return await asyncio.gather(*(_check_provider(model) for model in models))


def _load_cache(path: Path, models) -> dict[str, Any]:
    if not path.exists():
        return {
            "model_specs": _cache_specs(models),
            "responses": {},
            "failures": {},
        }
    cache = json.loads(path.read_text(encoding="utf-8"))
    old_specs = cache.get("model_specs", {})
    for model_id, model_spec in _cache_specs(models).items():
        if model_id in old_specs and old_specs[model_id] != model_spec:
            print(f"Configured model ID changed for {model_id}; refreshing only that model's cached responses.")
            for question_responses in cache.get("responses", {}).values():
                question_responses.pop(model_id, None)
            for question_failures in cache.get("failures", {}).values():
                question_failures.pop(model_id, None)
        old_specs[model_id] = model_spec
    cache["model_specs"] = old_specs
    cache.setdefault("responses", {})
    cache.setdefault("failures", {})
    return cache


def _question_complete(
    cache: dict[str, Any], question: dict[str, Any], model_ids: list[str]
) -> bool:
    responses = cache["responses"].get(question["id"], {})
    return all(
        model_id in responses
        and extract_choice(responses[model_id].get("answer"), question["choices"]) is not None
        for model_id in model_ids
    )


async def _collect_responses(questions, models, cache_path: Path, cache: dict[str, Any]):
    model_ids = [model.id for model in models]
    quota_blocked: dict[str, str] = {}
    for index, question in enumerate(questions, 1):
        question_id = question["id"]
        cache["responses"].setdefault(question_id, {})
        cache["failures"].setdefault(question_id, {})
        missing = [
            model for model in models
            if not cache["responses"][question_id].get(model.id, {}).get("answer")
        ]
        for model in missing:
            if model.id in quota_blocked:
                cache["failures"][question_id][model.id] = {
                    "error": quota_blocked[model.id],
                    "status": "not_attempted_daily_quota",
                }
        if missing and quota_blocked:
            _atomic_json(cache_path, cache)
        missing = [model for model in missing if model.id not in quota_blocked]
        if missing:
            prompt, choice_prompt = format_arc_prompt(question)
            tasks = {
                asyncio.create_task(call_model(model, prompt, choice_prompt=choice_prompt)): model
                for model in missing
            }
            for task in asyncio.as_completed(tasks):
                result = await task
                model_id = result["model_id"]
                choice = extract_choice(result.get("answer"), question["choices"])
                if choice is not None:
                    cache["responses"][question_id][model_id] = {
                        "answer": choice,
                        "confidence": result.get("confidence"),
                        "latency_ms": result.get("latency_ms"),
                        "raw_response": result.get("raw_response"),
                    }
                    cache["failures"][question_id].pop(model_id, None)
                else:
                    error = result.get("error") or "Provider returned no valid ARC choice."
                    cache["failures"][question_id][model_id] = {
                        "error": error,
                        "status": "failed",
                        "answer": result.get("answer"),
                        "raw_response": result.get("raw_response"),
                    }
                    if "daily quota exceeded" in error.lower():
                        quota_blocked[model_id] = error
                _atomic_json(cache_path, cache)
        if index % 10 == 0 or index == len(questions):
            completed = sum(
                _question_complete(cache, item, model_ids) for item in questions
            )
            print(f"Response progress: {completed}/{len(questions)} questions complete.")
    return cache


def _prepare_samples(questions, split_ids, cache, model_ids):
    question_by_id = {item["id"]: item for item in questions}
    split_data: dict[str, list[dict[str, Any]]] = {}
    for split_name, ids in split_ids.items():
        rows = []
        for question_id in ids:
            question = question_by_id[question_id]
            cached = cache["responses"].get(question_id, {})
            responses = [
                {"model_id": model_id, **cached[model_id]}
                for model_id in model_ids
                if model_id in cached and cached[model_id].get("answer")
            ]
            if len(responses) != len(model_ids):
                continue
            for response in responses:
                response["choice"] = extract_choice(response["answer"], question["choices"])
            invalid_responses = [
                response for response in responses if response["choice"] is None
            ]
            if invalid_responses:
                for response in invalid_responses:
                    cache.setdefault("failures", {}).setdefault(question_id, {})[
                        response["model_id"]
                    ] = {
                        "status": "failed",
                        "error": "Cached response is not a valid ARC choice label.",
                        "answer": response["answer"],
                        "raw_response": response.get("raw_response"),
                    }
                continue
            for response in responses:
                response["correct"] = response["choice"] == question["answer_key"].upper()
            grouped = group_answers(responses, model_ids)
            consensus_choice = extract_choice(grouped["consensus_answer"], question["choices"])
            rows.append({
                "id": question_id,
                "responses": responses,
                "grouping": grouped,
                "consensus_correct": consensus_choice == question["answer_key"].upper(),
                "consensus_choice": consensus_choice,
            })
        split_data[split_name] = rows
    return split_data


def _safe_imputer_classifier():
    # Missing provider confidence stays missing in raw records; the indicator preserves it for training.
    return make_pipeline(
        SimpleImputer(
            strategy="constant",
            fill_value=0.0,
            add_indicator=True,
            keep_empty_features=True,
        ),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )


def _score(model, row: list[float]) -> float:
    probabilities = model.predict_proba(np.asarray([row], dtype=float))[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(1)])


def _reliability(correct: list[bool], scores: list[float | None], bins: int = 10):
    valid = [(bool(target), float(score)) for target, score in zip(correct, scores) if score is not None]
    result = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            (target, score) for target, score in valid
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        result.append({
            "bin_center": (lower + upper) / 2,
            "mean_confidence": float(np.mean([score for _, score in selected])) if selected else None,
            "accuracy": float(np.mean([target for target, _ in selected])) if selected else None,
            "count": len(selected),
        })
    return result


def _histogram(scores: list[float | None], bins: int = 20):
    values = [float(score) for score in scores if score is not None]
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return {
        "bin_centers": [float((edges[i] + edges[i + 1]) / 2) for i in range(bins)],
        "counts": [int(count) for count in counts],
    }


def _system_metrics(correct: list[bool], scores: list[float | None], threshold: float):
    answered = [
        index for index, score in enumerate(scores)
        if score is not None and score >= threshold
    ]
    n_total = len(correct)
    n_answered = len(answered)
    selective_accuracy = (
        float(np.mean([correct[index] for index in answered])) if answered else None
    )
    valid_pairs = [
        (bool(target), float(score)) for target, score in zip(correct, scores)
        if score is not None
    ]
    brier = (
        float(np.mean([(score - float(target)) ** 2 for target, score in valid_pairs]))
        if valid_pairs else None
    )
    ece = None
    if valid_pairs:
        targets = np.asarray([target for target, _ in valid_pairs], dtype=int)
        confidences = np.asarray([score for _, score in valid_pairs], dtype=float)
        confidence_predictions = np.ones_like(targets)
        ece = compute_ece(targets, confidence_predictions, confidences, n_bins=10)
    accuracy = (
        float(sum(bool(correct[index]) for index in answered) / n_total)
        if n_total else None
    )
    return {
        "accuracy": accuracy,
        "selective_accuracy": selective_accuracy,
        "coverage": n_answered / n_total if n_total else 0.0,
        "abstention_rate": (n_total - n_answered) / n_total if n_total else 0.0,
        "ece": ece,
        "brier_score": brier,
        "answered": n_answered,
        "total": n_total,
    }


def _risk_coverage(correct: list[bool], scores: list[float | None]):
    curve = []
    thresholds = np.linspace(0.0, 1.0, 101)
    for threshold in thresholds:
        metrics = _system_metrics(correct, scores, float(threshold))
        risk = (
            1.0 - metrics["selective_accuracy"]
            if metrics["selective_accuracy"] is not None else None
        )
        curve.append({
            "threshold": float(threshold),
            "coverage": metrics["coverage"],
            "risk": risk,
        })
    points_by_coverage = {
        item["coverage"]: item["risk"] for item in curve if item["risk"] is not None
    }
    points = sorted(points_by_coverage.items())
    if not points:
        return curve, None
    if len(points) == 1:
        return curve, float(points[0][0] * points[0][1])
    if points[0][0] > 0:
        points.insert(0, (0.0, points[0][1]))
    aurc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        aurc += (x1 - x0) * (y0 + y1) / 2
    return curve, float(aurc)


def run_evaluation(*, cache_only: bool | None = None) -> dict[str, Any]:
    if cache_only is None:
        cache_only = os.getenv("TRUSTMETER_CACHE_ONLY", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
    models = load_models()
    if not 2 <= len(models) <= 4:
        raise ValueError(f"Configure 2 to 4 models; found {len(models)}.")
    questions = load_arc_questions()
    if not questions:
        raise RuntimeError("The ARC-Challenge test split returned no questions.")
    question_ids = [item["id"] for item in questions]
    split_ids = split_question_ids(question_ids)
    model_ids = [model.id for model in models]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "responses_cache.json"
    cache = _load_cache(cache_path, models)
    dataset_info = {
        "name": "ARC-Challenge",
        "source": "Hugging Face datasets: allenai/ai2_arc, ARC-Challenge configuration, test split",
        "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
        "sample_size": len(questions),
        "random_seed": SEED,
        "split_sizes": {name: len(ids) for name, ids in split_ids.items()},
        "features": ["question text", "multiple-choice option labels/text", "answerKey"],
        "split_question_ids": split_ids,
    }
    _atomic_json(DATA_DIR / "dataset_info.json", dataset_info)
    if not cache_only:
        missing_keys = [
            model.api_key_env for model in models
            if model.requires_api_key and not model.api_key
        ]
        if missing_keys:
            names = ", ".join(sorted(set(missing_keys)))
            raise RuntimeError(
                f"Missing required provider key(s): {names}. "
                "Dataset and split metadata were saved; no model questions were submitted."
            )
        unreachable = [
            name for name, reachable in asyncio.run(_check_providers(models))
            if not reachable
        ]
        if unreachable:
            raise RuntimeError(
                "Providers failed the reachability preflight: "
                + ", ".join(unreachable)
                + ". Dataset and split metadata were saved; no ARC questions were submitted."
            )
        asyncio.run(_collect_responses(questions, models, cache_path, cache))
    cache.setdefault("failures", {})
    complete = sum(
        _question_complete(cache, question, model_ids)
        for question in questions
    )
    question_by_id = {question["id"]: question for question in questions}
    complete_pairs = sum(
        extract_choice(
            cache["responses"].get(question_id, {}).get(model_id, {}).get("answer"),
            question_by_id[question_id]["choices"],
        ) is not None
        for question_id in question_ids for model_id in model_ids
    )
    expected_pairs = len(question_ids) * len(model_ids)
    split_data = _prepare_samples(questions, split_ids, cache, model_ids)
    _atomic_json(cache_path, cache)
    train_rows = split_data["train"]
    validation_rows = split_data["validation"]
    test_rows = split_data["test"]
    qualified_questions = sum(len(rows) for rows in split_data.values())
    model_response_counts = {
        model_id: {
            "successful_responses": sum(
                extract_choice(
                    cache["responses"].get(question_id, {}).get(model_id, {}).get("answer"),
                    question_by_id[question_id]["choices"],
                ) is not None
                for question_id in question_ids
            ),
            "failed_responses": sum(
                cache["failures"].get(question_id, {}).get(model_id, {}).get("status") == "failed"
                for question_id in question_ids
            ),
            "not_attempted_due_to_daily_quota": sum(
                cache["failures"].get(question_id, {}).get(model_id, {}).get("status")
                == "not_attempted_daily_quota"
                for question_id in question_ids
            ),
        }
        for model_id in model_ids
    }
    failure_count = sum(
        counts["failed_responses"] for counts in model_response_counts.values()
    )
    if qualified_questions != len(questions):
        print(
            f"Qualified questions: {qualified_questions}/{len(questions)}; "
            f"usable model responses: {complete_pairs}/{expected_pairs}; "
            f"recorded failures: {failure_count}."
        )
    if not train_rows or not validation_rows or not test_rows:
        raise RuntimeError(
            "Not enough fully answered questions to populate all splits. "
            "Successful responses and failures remain recorded in the cache."
        )

    historical_accuracy = {
        model_id: float(np.mean([
            next(response["correct"] for response in row["responses"] if response["model_id"] == model_id)
            for row in train_rows
        ]))
        for model_id in model_ids
    }
    feature_names = trust_feature_names(model_ids)
    feature = lambda row: calculate_trust_features(
        row["responses"], model_ids, historical_accuracy
    )
    train_x = np.asarray([feature(row) for row in train_rows], dtype=float)
    validation_x = np.asarray([feature(row) for row in validation_rows], dtype=float)
    test_x = np.asarray([feature(row) for row in test_rows], dtype=float)
    train_y = np.asarray([row["consensus_correct"] for row in train_rows], dtype=int)
    validation_y = np.asarray([row["consensus_correct"] for row in validation_rows], dtype=int)
    test_y = [bool(row["consensus_correct"]) for row in test_rows]
    if len(np.unique(train_y)) != 2 or len(np.unique(validation_y)) != 2:
        raise RuntimeError(
            "Trust calibration needs both correct and incorrect consensus examples in train and validation."
        )

    trust_base = _safe_imputer_classifier()
    trust_base.fit(train_x, train_y)
    trust_estimator = SigmoidCalibratedClassifier(trust_base).fit(validation_x, validation_y)
    trust_test_scores = [
        _score(trust_estimator, row.tolist()) for row in test_x
    ]

    # Select the best individual model using TRAIN accuracy only.
    best_model_id = max(
        model_ids,
        key=lambda model_id: (
            historical_accuracy[model_id],
            -model_ids.index(model_id),
        ),
    )
    per_model = {}
    single_calibrators = {}
    calibrated_model_scores = {}
    for model_id in model_ids:
        def confidence_feature(row):
            response = next(item for item in row["responses"] if item["model_id"] == model_id)
            value = response.get("confidence")
            return [float(value) / 100.0 if value is not None else float("nan")]

        model_train_x = np.asarray([confidence_feature(row) for row in train_rows])
        model_val_x = np.asarray([confidence_feature(row) for row in validation_rows])
        model_test_x = np.asarray([confidence_feature(row) for row in test_rows])
        model_train_y = np.asarray([
            next(item["correct"] for item in row["responses"] if item["model_id"] == model_id)
            for row in train_rows
        ], dtype=int)
        model_val_y = np.asarray([
            next(item["correct"] for item in row["responses"] if item["model_id"] == model_id)
            for row in validation_rows
        ], dtype=int)
        calibrated_scores: list[float | None]
        if len(np.unique(model_train_y)) == 2 and len(np.unique(model_val_y)) == 2:
            single_base = _safe_imputer_classifier()
            single_base.fit(model_train_x, model_train_y)
            single_calibrated = SigmoidCalibratedClassifier(single_base).fit(model_val_x, model_val_y)
            single_calibrators[model_id] = single_calibrated
            calibrated_scores = [_score(single_calibrated, row.tolist()) for row in model_test_x]
        else:
            calibrated_scores = [None] * len(test_rows)
        calibrated_model_scores[model_id] = calibrated_scores
        test_responses = [
            next(item for item in row["responses"] if item["model_id"] == model_id)
            for row in test_rows
        ]
        raw_scores = [
            float(item["confidence"]) / 100.0 if item.get("confidence") is not None else None
            for item in test_responses
        ]
        model_correct = [bool(item["correct"]) for item in test_responses]
        per_model[model_id] = {
            "display_name": next(model.display_name for model in models if model.id == model_id),
            "train_accuracy": historical_accuracy[model_id],
            "raw": _system_metrics(model_correct, raw_scores, DEFAULT_THRESHOLD),
            "calibrated": _system_metrics(model_correct, calibrated_scores, DEFAULT_THRESHOLD),
            "raw_reliability": _reliability(model_correct, raw_scores),
            "calibrated_reliability": _reliability(model_correct, calibrated_scores),
            "raw_confidence_distribution": _histogram(raw_scores),
            "calibrated_confidence_distribution": _histogram(calibrated_scores),
            "mean_latency_ms": float(np.mean([
                item["latency_ms"] for item in test_responses if item.get("latency_ms") is not None
            ])) if any(item.get("latency_ms") is not None for item in test_responses) else None,
            "confidence_available": sum(score is not None for score in raw_scores),
        }

    test_single = [
        next(item for item in row["responses"] if item["model_id"] == best_model_id)
        for row in test_rows
    ]
    raw_single_scores = [
        float(item["confidence"]) / 100.0 if item.get("confidence") is not None else None
        for item in test_single
    ]
    calibrated_single_score_series = calibrated_model_scores[best_model_id]
    best_single_correct = [bool(item["correct"]) for item in test_single]

    majority_correct = [bool(row["consensus_correct"]) for row in test_rows]
    majority_scores = [float(row["grouping"]["agreement_fraction"]) for row in test_rows]
    systems_series = {
        "best_single_raw_confidence": (best_single_correct, raw_single_scores),
        "best_single_calibrated_confidence": (best_single_correct, calibrated_single_score_series),
        "majority_vote": (majority_correct, majority_scores),
        "trustmeter": (test_y, trust_test_scores),
    }
    systems = {}
    curves = {}
    aurc_values = {}
    for name, (correct, scores) in systems_series.items():
        use_threshold = DEFAULT_THRESHOLD if name != "majority_vote" else 0.0
        systems[name] = _system_metrics(correct, scores, use_threshold)
        curves[name], aurc_values[name] = _risk_coverage(correct, scores)
        systems[name]["aurc"] = aurc_values[name]
    systems["best_single_raw_confidence"]["selected_model_id"] = best_model_id
    systems["best_single_calibrated_confidence"]["selected_model_id"] = best_model_id
    systems["majority_vote"]["threshold"] = None
    systems["trustmeter"]["threshold"] = DEFAULT_THRESHOLD
    trust_aurc = systems["trustmeter"]["aurc"]
    raw_aurc = systems["best_single_raw_confidence"]["aurc"]
    calibrated_aurc = systems["best_single_calibrated_confidence"]["aurc"]
    hypothesis_supported = (
        trust_aurc < raw_aurc and trust_aurc < calibrated_aurc
        if trust_aurc is not None and raw_aurc is not None and calibrated_aurc is not None
        else None
    )

    trust_reliability = _reliability(test_y, trust_test_scores)
    trust_histogram = _histogram(trust_test_scores)
    trust_curve = curves["trustmeter"]
    metrics = {
        "dataset_info": {
            key: value for key, value in dataset_info.items()
            if key != "split_question_ids"
        },
        "evaluation": {
            "split_sizes": dataset_info["split_sizes"],
            "qualified_split_sizes": {
                name: len(rows) for name, rows in split_data.items()
            },
            "sample_size": len(questions),
            "qualified_questions": qualified_questions,
            "fully_cached_questions": complete,
            "incomplete_questions": len(questions) - qualified_questions,
            "usable_model_responses": complete_pairs,
            "expected_model_responses": expected_pairs,
            "recorded_failures": failure_count,
            "model_response_counts": model_response_counts,
            "test_set_size": len(test_rows),
            "random_seed": SEED,
            "threshold": DEFAULT_THRESHOLD,
            "systems_evaluated_on": "test split only",
            "response_source": "cache_only" if cache_only else "providers_and_cache",
        },
        "best_single_model": {
            "id": best_model_id,
            "display_name": per_model[best_model_id]["display_name"],
            "selection_basis": "highest per-model accuracy on the train split",
        },
        "historical_accuracy": historical_accuracy,
        "per_model": per_model,
        "systems": systems,
        "hypothesis_assessment": {
            "criterion": "TrustMeter test AURC is lower than both best-single raw- and calibrated-confidence test AURC",
            "supported": hypothesis_supported,
        },
        "risk_coverage": curves,
        "reliability": {
            "trustmeter": trust_reliability,
            "per_model": {
                model_id: {
                    "raw": per_model[model_id]["raw_reliability"],
                    "calibrated": per_model[model_id]["calibrated_reliability"],
                }
                for model_id in model_ids
            },
        },
        "confidence_distribution": {
            "trustmeter": trust_histogram,
            "per_model": {
                model_id: {
                    "raw": per_model[model_id]["raw_confidence_distribution"],
                    "calibrated": per_model[model_id]["calibrated_confidence_distribution"],
                }
                for model_id in model_ids
            },
        },
        "limitations": [
            "The sample is small and drawn from ARC-Challenge multiple-choice questions.",
            "Provider models may change and free-tier rate limits can affect availability.",
            "Live free-form answers are grouped with normalized string matching, not semantic equivalence.",
            "Live verdicts estimate correctness; ARC historical accuracy may not transfer to other topics.",
        ],
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trust_estimator, MODELS_DIR / "trust_estimator.joblib")
    joblib.dump(single_calibrators, MODELS_DIR / "single_calibrators.joblib")
    _atomic_json(MODELS_DIR / "training_metadata.json", {
        "model_ids": model_ids,
        "feature_names": feature_names,
        "historical_accuracy": historical_accuracy,
        "best_single_model_id": best_model_id,
    })
    _atomic_json(MODELS_DIR / "metrics.json", metrics)
    print(f"Completed ARC evaluation: {len(train_rows)} train, {len(validation_rows)} validation, {len(test_rows)} test.")
    print(f"Qualified questions: {qualified_questions}/{len(questions)}.")
    print(json.dumps(systems, indent=2))
    return metrics


def main() -> int:
    try:
        run_evaluation()
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
