from app.trust import calculate_trust_features, group_answers, normalize_answer
import math


MODEL_IDS = ["one", "two", "three"]


def test_normalization_strips_case_punctuation_and_articles():
    assert normalize_answer("The quick, brown fox!") == "quick brown fox"


def test_plurality_agreement_and_dissent_are_reported():
    responses = [
        {"model_id": "one", "answer": "The blue whale"},
        {"model_id": "two", "answer": "blue whale!"},
        {"model_id": "three", "answer": "A dolphin"},
    ]

    result = group_answers(responses, MODEL_IDS)

    assert result["consensus_answer"] == "The blue whale"
    assert result["supporting_model_ids"] == ["one", "two"]
    assert result["dissenting_model_ids"] == ["three"]
    assert result["agreement_fraction"] == 2 / 3


def test_live_agreement_uses_only_models_that_returned_answers():
    result = group_answers(
        [{"model_id": "one", "answer": "A"}, {"model_id": "two", "answer": None}],
        MODEL_IDS,
    )

    assert result["agreement_fraction"] == 1.0
    assert result["supporting_model_ids"] == ["one"]
    assert result["dissenting_model_ids"] == []


def test_trust_features_preserve_missing_confidence_and_use_only_supporters():
    responses = [
        {"model_id": "one", "answer": "A", "confidence": 80},
        {"model_id": "two", "answer": "a", "confidence": None},
        {"model_id": "three", "answer": "B", "confidence": 60},
    ]
    historical = {"one": 0.8, "two": 0.7, "three": 0.9}

    features = calculate_trust_features(responses, MODEL_IDS, historical)

    assert features[0] == 2 / 3
    assert features[1] == 0.8
    assert features[2] == 0.75
    assert features[3:6] == [0.8, 1.0, 0.8]
    assert math.isnan(features[6])
    assert features[7:9] == [1.0, 0.7]
    assert features[9:11] == [0.6, 0.0]
    assert math.isnan(features[11])
