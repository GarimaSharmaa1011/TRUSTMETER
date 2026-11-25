
"""TrustMeter multi-model question-checking API."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import BACKEND_DIR, ModelConfig, load_models
from app.providers import ask_models
from app.trust import calculate_trust_features, group_answers, trust_feature_names

logger = logging.getLogger("trustmeter")
MODELS = load_models()
MODEL_IDS = [model.id for model in MODELS]
MODEL_BY_ID = {model.id: model for model in MODELS}
DEFAULT_THRESHOLD = float(os.getenv("TRUST_THRESHOLD", "0.70"))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank.")
        return value


class BatchAskRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=20)
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("questions")
    @classmethod
    def questions_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("Questions must not contain blank entries.")
        if any(len(value) > 4000 for value in cleaned):
            raise ValueError("Each question must be at most 4000 characters.")
        return cleaned


state: dict[str, Any] = {
    "trust_estimator": None,
    "historical_accuracy": {},
    "metrics": None,
    "error": None,
}


def load_saved_evaluation() -> None:
    model_path = BACKEND_DIR / "models" / "trust_estimator.joblib"
    metadata_path = BACKEND_DIR / "models" / "training_metadata.json"
    metrics_path = BACKEND_DIR / "models" / "metrics.json"

    try:
        if model_path.exists() and metadata_path.exists() and metrics_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            if (
                metadata.get("model_ids") != MODEL_IDS
                or metadata.get("feature_names") != trust_feature_names(MODEL_IDS)
            ):
                state["trust_estimator"] = None
                state["metrics"] = None
                state["error"] = (
                    "Configured models differ from the saved evaluation. "
                    "Rerun ARC evaluation."
                )
                return

            state["trust_estimator"] = joblib.load(model_path)
            state["historical_accuracy"] = metadata["historical_accuracy"]
            state["metrics"] = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            state["error"] = None

        else:
            state["trust_estimator"] = None
            state["metrics"] = None
            state["error"] = (
                "ARC evaluation has not been completed. "
                "Run python -m train.evaluate."
            )

    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.exception("Could not load saved ARC evaluation.")
        state["trust_estimator"] = None
        state["metrics"] = None
        state["error"] = (
            f"Saved evaluation could not be loaded ({type(exc).__name__})."
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_saved_evaluation()
    yield


app = FastAPI(
    title="TrustMeter — When Should an AI Admit It Is Unsure?",
    description=(
        "Multi-model answer comparison with a calibrated trust estimate. "
        "A verdict is not proof of correctness."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# CORS: allow local development and the deployed Vercel frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://trustmeter-ochre.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_response(request: Request, exc: RequestValidationError):
    del request

    errors = [
        {
            "loc": list(error["loc"]),
            "msg": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]

    return JSONResponse(status_code=422, content={"detail": errors})


async def _provider_status(model: ModelConfig) -> dict[str, Any]:
    if not model.api_key:
        return {
            "id": model.id,
            "display_name": model.display_name,
            "configured": False,
            "reachable": False,
            "error": f"API key is not configured ({model.api_key_env}).",
        }

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                model.base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {model.api_key}"},
            )

        return {
            "id": model.id,
            "display_name": model.display_name,
            "configured": True,
            "reachable": response.status_code < 400 or response.status_code == 429,
            "status_code": response.status_code,
            "error": (
                f"Provider returned HTTP {response.status_code}."
                if response.status_code >= 400
                else None
            ),
        }

    except httpx.TimeoutException:
        return {
            "id": model.id,
            "display_name": model.display_name,
            "configured": True,
            "reachable": False,
            "error": "Provider health check timed out.",
        }

    except httpx.HTTPError as exc:
        return {
            "id": model.id,
            "display_name": model.display_name,
            "configured": True,
            "reachable": False,
            "error": f"Provider health check failed ({type(exc).__name__}).",
        }


async def _answer(question: str, threshold: float) -> dict[str, Any]:
    responses = await ask_models(MODELS, question)
    grouping = group_answers(responses, MODEL_IDS)

    answered_model_count = sum(
        bool(item.get("answer")) for item in responses
    )

    trust_score: float | None = None

    if state["trust_estimator"] is not None and grouping["consensus_answer"]:
        features = calculate_trust_features(
            responses,
            MODEL_IDS,
            state["historical_accuracy"],
        )

        probabilities = state["trust_estimator"].predict_proba(
            np.asarray([features], dtype=float)
        )[0]

        classes = list(state["trust_estimator"].classes_)

        if 1 in classes:
            trust_score = float(probabilities[classes.index(1)])

    provider_errors = [
        item for item in responses if item.get("error")
    ]

    if state["trust_estimator"] is None:
        status = "trust_estimator_not_trained"
        verdict = None
        message = (
            "Trust estimator not trained; raw model answers "
            "are shown without a trust verdict."
        )

    elif not grouping["consensus_answer"] or trust_score is None:
        status = "ok" if answered_model_count else "no_usable_answers"
        verdict = "ABSTAIN"
        message = "No model has a trustworthy answer."

    elif (
        grouping["tie"]
        or grouping["dissenting_model_ids"]
        or provider_errors
    ):
        status = "ok"
        verdict = "DISPUTED"
        message = (
            "Models disagree or some providers failed; "
            "treat the consensus as disputed."
        )

    elif trust_score < threshold:
        status = "ok"
        verdict = "ABSTAIN"
        message = "No model has a trustworthy answer."

    else:
        status = "ok"
        verdict = "TRUSTED"
        message = (
            "Models agree and the calibrated trust estimate "
            "meets the threshold."
        )

    return {
        "question": question,
        "status": status,
        "answered_model_count": answered_model_count,
        "configured_model_count": len(MODEL_IDS),
        "models": [
            {
                "model_id": item["model_id"],
                "display_name": MODEL_BY_ID[item["model_id"]].display_name,
                "answer": item.get("answer"),
                "confidence": item.get("confidence"),
                "latency_ms": item.get("latency_ms"),
                "error": item.get("error"),
            }
            for item in responses
        ],
        "consensus_answer": grouping["consensus_answer"],
        "agreement_fraction": grouping["agreement_fraction"],
        "trust_score": trust_score,
        "threshold": threshold,
        "verdict": verdict,
        "supporting_models": [
            MODEL_BY_ID[model_id].display_name
            for model_id in grouping["supporting_model_ids"]
        ],
        "dissenting_models": [
            MODEL_BY_ID[model_id].display_name
            for model_id in grouping["dissenting_model_ids"]
        ],
        "message": message,
        "limitations": [
            "Live verdicts estimate correctness; they are not proof.",
            "Agreement uses normalized string matching, not semantic equivalence.",
            "Historical model accuracy comes from ARC-Challenge and may not transfer to other topics.",
            f"Trust was computed from {answered_model_count} of {len(MODEL_IDS)} configured models.",
        ],
    }


@app.get("/health")
async def health():
    if os.getenv(
        "TRUSTMETER_DISABLE_PROVIDER_HEALTH_CHECKS", ""
    ).strip().lower() in {"1", "true", "yes"}:

        providers = [
            {
                "id": model.id,
                "display_name": model.display_name,
                "configured": bool(model.api_key),
                "reachable": None,
                "error": (
                    "Provider reachability was not checked (offline mode)."
                ),
            }
            for model in MODELS
        ]

    else:
        providers = await asyncio.gather(
            *(_provider_status(model) for model in MODELS)
        )

    return {
        "status": "ok",
        "trust_estimator_loaded": state["trust_estimator"] is not None,
        "evaluation_available": state["metrics"] is not None,
        "evaluation_error": state["error"],
        "providers": providers,
    }


@app.get("/metrics")
def get_metrics():
    if state["metrics"] is None:
        raise HTTPException(status_code=503, detail=state["error"])

    return state["metrics"]


@app.get("/reliability")
def get_reliability():
    if state["metrics"] is None:
        raise HTTPException(status_code=503, detail=state["error"])

    metrics = state["metrics"]

    return {
        "reliability": metrics["reliability"],
        "risk_coverage": metrics["risk_coverage"],
        "confidence_distribution": metrics["confidence_distribution"],
    }


@app.post("/ask")
async def ask(request: AskRequest):
    return await _answer(request.question, request.threshold)


@app.post("/ask/batch")
async def ask_batch(request: BatchAskRequest):
    results = []

    for question in request.questions:
        results.append(
            await _answer(question, request.threshold)
        )

    return {
        "results": results,
        "count": len(results),
    }
