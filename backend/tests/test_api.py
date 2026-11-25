from fastapi.testclient import TestClient

import app.main as api


class FixedTrustEstimator:
    classes_ = [0, 1]

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, _features):
        return [[1 - self.probability, self.probability]]


def test_ask_abstains_below_threshold_and_names_supporting_models(monkeypatch):
    async def fake_ask_models(models, question):
        return [
            {"model_id": model.id, "answer": "42", "confidence": 90, "latency_ms": 1, "error": None}
            for model in models
        ]

    monkeypatch.setattr(api, "ask_models", fake_ask_models)
    with TestClient(api.app) as client:
        api.state["trust_estimator"] = FixedTrustEstimator(0.69)
        api.state["historical_accuracy"] = {model_id: 0.7 for model_id in api.MODEL_IDS}
        response = client.post("/ask", json={"question": "What is the answer?", "threshold": 0.7})
        assert response.status_code == 200
        assert response.json()["verdict"] == "ABSTAIN"
        assert response.json()["message"].startswith("No model has a trustworthy answer.")

        api.state["trust_estimator"] = FixedTrustEstimator(0.95)
        response = client.post("/ask", json={"question": "What is the answer?", "threshold": 0.7})
        assert response.status_code == 200
        assert response.json()["verdict"] == "TRUSTED"
        assert len(response.json()["supporting_models"]) == len(api.MODEL_IDS)


def test_ask_validation_rejects_empty_question_and_invalid_threshold():
    with TestClient(api.app) as client:
        assert client.post("/ask", json={"question": "   "}).status_code == 422
        assert client.post("/ask", json={"question": "valid", "threshold": 2}).status_code == 422
        assert client.post("/ask/batch", json={"questions": []}).status_code == 422
        assert client.post("/predict", json={}).status_code == 404


def test_ask_without_estimator_returns_raw_answers_without_verdict(monkeypatch):
    async def fake_ask_models(models, question):
        return [
            {"model_id": model.id, "answer": "Au", "confidence": 95, "latency_ms": 1, "error": None}
            for model in models
        ]

    monkeypatch.setattr(api, "ask_models", fake_ask_models)
    with TestClient(api.app) as client:
        api.state["trust_estimator"] = None
        response = client.post("/ask", json={"question": "Symbol for gold?"})

    assert response.status_code == 200
    assert response.json()["status"] == "trust_estimator_not_trained"
    assert response.json()["verdict"] is None
    assert response.json()["answered_model_count"] == len(api.MODEL_IDS)
    assert response.json()["models"][0]["answer"] == "Au"


def test_ask_with_provider_failure_reports_answer_count_and_dispute(monkeypatch):
    async def fake_ask_models(models, question):
        return [
            {
                "model_id": model.id,
                "answer": "Au" if index == 0 else None,
                "confidence": 99 if index == 0 else None,
                "latency_ms": 1,
                "error": None if index == 0 else "Provider request failed.",
            }
            for index, model in enumerate(models)
        ]

    monkeypatch.setattr(api, "ask_models", fake_ask_models)
    with TestClient(api.app) as client:
        api.state["trust_estimator"] = FixedTrustEstimator(0.95)
        api.state["historical_accuracy"] = {model_id: 0.8 for model_id in api.MODEL_IDS}
        response = client.post("/ask", json={"question": "Symbol for gold?"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["verdict"] == "DISPUTED"
    assert response.json()["answered_model_count"] == 1
    assert response.json()["configured_model_count"] == len(api.MODEL_IDS)
    assert response.json()["models"][1]["error"] == "Provider request failed."


def test_ask_with_disagreement_reports_disputed(monkeypatch):
    async def fake_ask_models(models, question):
        return [
            {
                "model_id": model.id,
                "answer": "A" if index == 0 else "B",
                "confidence": 95,
                "latency_ms": 1,
                "error": None,
            }
            for index, model in enumerate(models)
        ]

    monkeypatch.setattr(api, "ask_models", fake_ask_models)
    with TestClient(api.app) as client:
        api.state["trust_estimator"] = FixedTrustEstimator(0.5)
        api.state["historical_accuracy"] = {model_id: 0.8 for model_id in api.MODEL_IDS}
        response = client.post("/ask", json={"question": "Pick an answer?"})

    assert response.status_code == 200
    assert response.json()["verdict"] == "DISPUTED"
    assert response.json()["answered_model_count"] == len(api.MODEL_IDS)


def test_health_metrics_reliability_and_batch_routes(monkeypatch):
    async def fake_provider_status(model):
        return {
            "id": model.id,
            "display_name": model.display_name,
            "configured": True,
            "reachable": True,
            "error": None,
        }

    async def fake_ask_models(models, question):
        return [
            {"model_id": model.id, "answer": "42", "confidence": 90, "latency_ms": 1, "error": None}
            for model in models
        ]

    monkeypatch.setattr(api, "_provider_status", fake_provider_status)
    monkeypatch.setattr(api, "ask_models", fake_ask_models)
    with TestClient(api.app) as client:
        api.state["trust_estimator"] = FixedTrustEstimator(0.9)
        api.state["historical_accuracy"] = {model_id: 0.8 for model_id in api.MODEL_IDS}
        api.state["metrics"] = {
            "reliability": {"trustmeter": []},
            "risk_coverage": {"trustmeter": []},
            "confidence_distribution": {"trustmeter": {}},
        }
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200
        assert client.get("/reliability").status_code == 200
        batch = client.post("/ask/batch", json={"questions": ["one?", "two?"]})
        assert batch.status_code == 200
        assert batch.json()["count"] == 2
        assert client.post("/ask/batch", json={"questions": ["ok", " "]}).status_code == 422
