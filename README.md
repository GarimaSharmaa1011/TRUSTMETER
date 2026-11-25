# TrustMeter — When Should an AI Admit It Is Unsure?

TrustMeter asks several configured language models the same question, groups answers using normalized string matching, and estimates whether the plurality answer is correct. Its verdict is an **estimate, not proof**. The estimator is trained and calibrated on ARC-Challenge; its historical accuracy is less certain on other subjects.

## What the system does

- Calls models through one OpenAI-compatible provider layer (`openai` Python package). Groq, Gemini-compatible endpoints, OpenRouter, and Ollama can be selected in `backend/config/models.yaml`.
- Displays each live answer, self-reported confidence when valid, latency, and provider error. A malformed/missing confidence remains `null`; it is not guessed.
- When a trained ARC estimator is available, returns **TRUSTED**, **DISPUTED**, or **ABSTAIN**. Without one, reports `trust_estimator_not_trained`, raw answers, and no verdict.
- Uses exact normalized string matching (lowercase, punctuation/article stripping) to group live free-form answers. Paraphrases may be treated as disagreement; semantic matching is not enabled.
- Records ARC-Challenge responses and split information, then trains and evaluates a trust estimator without using validation or test labels for training.
- Saves held-out evaluation results to `backend/models/metrics.json`. The dashboard does not show placeholder or fabricated results if evaluation has not completed.

## Requirements

- Python 3.10+
- Node.js 18+ and npm
- API credentials for the hosted providers you choose, and/or a running Ollama server with the configured model installed

## Provider setup

From the project root in PowerShell:

```powershell
Copy-Item .\backend\.env.example .\backend\.env
```

Edit `backend\.env` locally and add your provider key(s); do not paste keys into source files or commit `.env`. The default live model set is Groq GPT-OSS 20B and 120B. Gemini Flash is configured as optional and disabled by default because its free-tier daily request quota was exhausted during evaluation. Its existing 22 cached responses are preserved. To explicitly include it after its quota resets, set `TRUSTMETER_INCLUDE_OPTIONAL_MODELS=gemini-flash` in the shell that runs evaluation and the backend. A successful evaluation and its backend must use the same active model set.

Create credentials in the provider's account console:

- Groq models use `GROQ_API_KEY`.
- Gemini's OpenAI-compatible endpoint uses `GEMINI_API_KEY`. Gemini's observed 429 was a **per-day free-tier quota** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, limit 20), not a per-minute quota. Wait for the provider's reset before opting it back into evaluation.
- For OpenRouter, set that model's `base_url` to `https://openrouter.ai/api/v1`, use the OpenRouter model ID, and set `api_key_env` to an environment variable that you add to `backend/.env`.

The Groq `/models` catalog was checked before choosing alternatives. At that time, `qwen/qwen3.8-27b` was available but Groq listed it at $0.80/1M input and $4.00/1M output tokens, so it was not used without approval; `allam-2-7b` had pricing marked pending and was not treated as free. The available free catalog did not offer another suitable general-purpose chat model from a different family. Model IDs and provider prices/rate limits can change; verify them before adding a model. TrustMeter reports missing credentials, timeouts, rate limits, and provider errors rather than substituting answers.

`requests_per_minute` in `backend/config/models.yaml` configures a process-local limiter shared by models using the same provider URL and key variable. 429 responses use bounded exponential backoff; a detected per-day quota blocks further requests to that model for the remainder of that evaluation process.

## Install

Terminal 1:

```powershell
cd .\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Terminal 2:

```powershell
cd .\frontend
npm install
```

## Run a cache-only evaluation

From `backend` with the virtual environment active. This command uses only cached ARC data and the two cached Groq model responses; it disables Gemini and makes no provider requests:

```powershell
$env:ARC_SAMPLE_SIZE = "150"
$env:ARC_RANDOM_SEED = "42"
$env:TRUSTMETER_INCLUDE_OPTIONAL_MODELS = ""
$env:TRUSTMETER_CACHE_ONLY = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
python -m train.evaluate
```

The evaluator loads the locally cached ARC-Challenge test split, draws the seed-42 150-question sample, and keeps the original fixed 50% / 25% / 25% ID split. Cache-only mode skips provider-key checks, provider reachability checks, and response collection. Only questions where both active models have a valid ARC choice are included; invalid cached answers are retained and excluded. Configuration changes do not clear unrelated cache entries.

The real cache-only results are 148 qualified questions: 74 train, 36 validation, and 38 test. Two Groq outputs were not valid ARC labels and were not recalled. The evaluator saves the trust estimator, per-model calibrators, training metadata, and test-only metrics under `backend/models/`.

## Start the application

Terminal 1, from `backend`:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2, from `frontend`:

```powershell
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` requests to FastAPI on port 8000. If evaluation has not completed, live model answers can still be displayed, but no trust verdict is returned. If a provider key is missing or the provider is unavailable, that model displays an error.

To keep the health endpoint from contacting providers while running in offline mode, set `$env:TRUSTMETER_DISABLE_PROVIDER_HEALTH_CHECKS = "1"` before starting the backend. The UI will label provider reachability “Not checked.”

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Evaluation readiness and provider configuration/reachability |
| `GET` | `/metrics` | Saved test-split metrics |
| `GET` | `/reliability` | Saved reliability, confidence distribution, and risk/coverage data |
| `POST` | `/ask` | Ask all configured models and return responses, trust estimate, and verdict |
| `POST` | `/ask/batch` | Ask 1–20 questions sequentially |

Example:

```json
{
  "question": "Why do the seasons change?",
  "threshold": 0.7
}
```

Questions must contain non-whitespace text and be at most 4,000 characters. Threshold is in `[0, 1]`. Batch input is `{ "questions": ["Question 1?", "Question 2?"], "threshold": 0.7 }`.

If evaluation artifacts are missing, `/ask` returns the raw provider answers with `status: "trust_estimator_not_trained"` and `verdict: null`; it does not fabricate an `ABSTAIN` verdict. Responses also report the number of models that returned a usable answer. After training, trust features use only responding models and provider failures are shown. Model disagreement or a provider failure with a usable consensus is marked `DISPUTED`; when models agree, a trust score below the threshold is `ABSTAIN`.

## Tests and checks

From `backend`:

```powershell
python -m pytest
```

From `frontend`:

```powershell
npm run lint
npm run build
```

## Real results (150-question sample, seed 42)

Gemini was excluded from this run. The two Groq GPT-OSS models have 149 valid ARC-choice responses each; **148 questions** had valid answers from both models. The fixed split sizes are 74 train / 36 validation / 38 test. Metrics below are test-only, with `n=38` for each comparison system.

| Test system (n=38) | Accuracy | Selective accuracy | Coverage | ECE | Brier | AURC |
|---|---:|---:|---:|---:|---:|---:|
| Best single, raw confidence (GPT-OSS 120B) | 97.37% | 97.37% | 100.00% | 0.0503 | 0.0226 | 0.0014 |
| Best single, calibrated confidence (GPT-OSS 120B) | 92.11% | 100.00% | 92.11% | 0.0422 | 0.0292 | 0.0014 |
| Majority/agreement baseline | 97.37% | 97.37% | 100.00% | 0.0263 | 0.0263 | 0.0263 |
| TrustMeter with reject option | 97.37% | 97.37% | 100.00% | 0.0042 | 0.0256 | 0.0107 |

| Model (test n=38) | Raw accuracy | Raw ECE | Raw Brier | Calibrated accuracy | Calibrated ECE | Calibrated Brier |
|---|---:|---:|---:|---:|---:|---:|
| Groq GPT-OSS 20B | 92.11% | 0.0829 | 0.0848 | 97.37% | 0.0496 | 0.0302 |
| Groq GPT-OSS 120B | 97.37% | 0.0503 | 0.0226 | 92.11% | 0.0422 | 0.0292 |

The full 101-point test risk-coverage curves are in `backend/models/metrics.json`. By the predeclared criterion, the hypothesis is **not supported** on this sample: TrustMeter's test AURC (0.0107) is higher than both best-single AURCs (0.0014). The TrustMeter ECE is lower, but that does not satisfy the AURC criterion.

Previously recorded smoke checks outside the ARC evaluation: both active Groq models agreed and returned `TRUSTED` for gold (`Au`), the Red Planet (`Mars`), and the sheep riddle (`9`). The hot-dog and nonexistent-Blorvia questions returned `DISPUTED` because their compatible answers differed in wording under normalized string matching. All five had 2/2 usable model responses. These results were not re-queried during the cache-only verification; details and trust scores are in [docs/research.md](./docs/research.md).

## Limitations

The test set contains only 38 questions from a 150-question sample of ARC-Challenge, a multiple-choice benchmark. The two Groq models are both from the GPT-OSS family, so their errors may be correlated and agreement is not independent evidence. Gemini is excluded because its observed free-tier quota was 20 requests per day; no replacement Groq model was used because the available different-family chat candidate was metered or had pending pricing. Live free-form answer grouping uses normalized string matching, not semantic equivalence. Self-reported confidence and ARC calibration do not establish correctness on general questions.

See [docs/research.md](./docs/research.md) for the research design and the status of experimental results.
