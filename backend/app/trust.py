"""Answer grouping and trust-estimator feature construction."""

import re
import string
from typing import Any

ARTICLES = {"a", "an", "the"}


def normalize_answer(answer: str | None) -> str:
    if not answer:
        return ""
    text = answer.lower().translate(str.maketrans("", "", string.punctuation))
    words = re.sub(r"\s+", " ", text).split()
    normalized = [word for word in words if word not in ARTICLES]
    return " ".join(normalized or words)


def group_answers(
    responses: list[dict[str, Any]], model_ids: list[str]
) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    answer_by_model: dict[str, str] = {}
    for response in responses:
        model_id = response.get("model_id")
        normalized = normalize_answer(response.get("answer"))
        if model_id in model_ids and normalized:
            groups.setdefault(normalized, []).append(model_id)
            answer_by_model[model_id] = response["answer"].strip()
    if not groups:
        return {
            "consensus_answer": None,
            "normalized_answer": None,
            "supporting_model_ids": [],
            "dissenting_model_ids": [],
            "agreement_fraction": 0.0,
            "tie": False,
        }
    maximum = max(map(len, groups.values()))
    plurality = next(answer for answer, supporters in groups.items() if len(supporters) == maximum)
    supporters = groups[plurality]
    dissenters = [
        model_id for model_id in model_ids
        if model_id in answer_by_model and model_id not in supporters
    ]
    return {
        "consensus_answer": answer_by_model[supporters[0]],
        "normalized_answer": plurality,
        "supporting_model_ids": supporters,
        "dissenting_model_ids": dissenters,
        "agreement_fraction": len(supporters) / len(answer_by_model) if answer_by_model else 0.0,
        "tie": sum(len(ids) == maximum for ids in groups.values()) > 1,
    }


def trust_feature_names(model_ids: list[str]) -> list[str]:
    names = ["agreement_fraction", "mean_agree_confidence", "mean_agree_accuracy"]
    for model_id in model_ids:
        names.extend((
            f"confidence:{model_id}",
            f"agrees:{model_id}",
            f"agree_accuracy:{model_id}",
        ))
    return names


def calculate_trust_features(
    responses: list[dict[str, Any]],
    model_ids: list[str],
    historical_accuracy: dict[str, float | None],
) -> list[float]:
    grouped = group_answers(responses, model_ids)
    by_id = {response.get("model_id"): response for response in responses}
    supporters = set(grouped["supporting_model_ids"])
    agreeing_confidences = []
    agreeing_accuracy = []
    for model_id in supporters:
        confidence = by_id[model_id].get("confidence")
        if confidence is not None:
            agreeing_confidences.append(float(confidence) / 100.0)
        accuracy = historical_accuracy.get(model_id)
        if accuracy is not None:
            agreeing_accuracy.append(float(accuracy))
    features = [
        grouped["agreement_fraction"],
        sum(agreeing_confidences) / len(agreeing_confidences) if agreeing_confidences else float("nan"),
        sum(agreeing_accuracy) / len(agreeing_accuracy) if agreeing_accuracy else float("nan"),
    ]
    for model_id in model_ids:
        response = by_id.get(model_id, {})
        confidence = response.get("confidence")
        features.extend((
            float(confidence) / 100.0 if confidence is not None else float("nan"),
            float(model_id in supporters),
            float(historical_accuracy[model_id])
            if model_id in supporters and historical_accuracy.get(model_id) is not None
            else float("nan"),
        ))
    return features
