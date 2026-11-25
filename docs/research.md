# TrustMeter — When Should an AI Admit It Is Unsure?

## Problem statement

Language models can provide plausible but incorrect answers and can express unwarranted confidence. TrustMeter compares several model responses and estimates whether their plurality answer is correct, with an abstention option below a selected threshold. A live verdict is an **estimate of correctness, not proof**.

## Objectives

- Compare the highest-train-accuracy single model using its raw self-reported confidence and a calibrated confidence estimate.
- Compare those single-model systems with a no-abstention plurality vote and TrustMeter's agreement/calibrated-trust/reject system.
- Report accuracy, selective accuracy, coverage, abstention rate, ECE, Brier score, AURC, per-model results, and test risk-coverage curves.
- Make live provider failures and missing confidence explicit rather than filling them with fabricated outputs or scores.

## Hypothesis

> Combining multi-model agreement with calibrated confidence detects wrong answers better than any single model's confidence.

The predeclared comparison criterion in the saved evaluation is lower area under the risk-coverage curve (AURC) for TrustMeter than for both the train-selected best single model using raw confidence and that same model using calibrated confidence. A lower AURC is evidence on this sample only, not a general guarantee.

## Methodology

The evaluator loads the ARC-Challenge test split via Hugging Face `datasets`, chooses a deterministic sample (150 questions, seed 42 for this experiment), and partitions IDs into disjoint 50% train, 25% validation, and 25% test subsets. Each sampled question is sent to every active model. Successful outputs are cached per question/model to support resumption.

Each answer is graded by exact choice label against ARC's `answerKey`. The best single model is selected using train-split accuracy only. Train-split per-model accuracies are used as historical model features. The trust classifier is trained only on train examples; its sigmoid mapping is fitted only on validation examples. Per-model confidence calibrators follow the same train/validation separation. Final system and per-model metrics, reliability bins, confidence distributions, and risk-coverage curves use the test split only.

Live free-form answers are grouped by lowercase string normalization, punctuation stripping, and removal of English articles. This is not semantic equivalence: paraphrases can be counted as disagreement. Confidence is parsed from the provider's requested JSON and remains missing if invalid.

## Dataset

- **Name:** AI2 ARC-Challenge, test split.
- **Source:** [Hugging Face dataset `allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc), configuration `ARC-Challenge`.
- **Task:** multiple-choice question answering with answer labels and a known `answerKey`.
- **Sample and split details:** written to `backend/data/dataset_info.json` by a successful or attempted evaluation. The exact sample and split IDs are stored there.

## Provider set and quota handling

The original run used Groq GPT-OSS 20B, Groq GPT-OSS 120B, and Gemini Flash. A direct Gemini inference request returned HTTP 429 with `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quota value 20, and a provider-provided retry delay of about 18 hours 24 minutes. This is the **per-day** free-tier quota, not a per-minute limit. Gemini is now marked optional and disabled by default; its 22 cached responses are retained and were excluded from this run. The currently checked Groq catalog had no suitable different-family chat model confirmed free: `qwen/qwen3.8-27b` is active but priced at $0.80/1M input and $4.00/1M output tokens, while `allam-2-7b` had pricing listed as pending. Neither was used without cost approval. Re-enable Gemini explicitly after its daily quota resets with `TRUSTMETER_INCLUDE_OPTIONAL_MODELS=gemini-flash`.

Provider RPM limits are configurable in `backend/config/models.yaml`, shared between configured models on the same provider/key, and enforced client-side. 429 responses use bounded exponential backoff. A daily quota response is recorded as a model failure and prevents repeated requests to that model for the rest of the run. Cache entries are keyed by question and model; changing active models no longer deletes unrelated successful responses. Failed or invalid ARC-choice responses are recorded separately and are not treated as usable.

## Experimental results

**Cache-only evaluation completed.** No provider calls were made during this run. Gemini remained disabled; only cached Groq GPT-OSS 20B and 120B outputs were used. Each had 149 valid ARC choices out of 150 sampled questions; the two invalid outputs were retained and recorded as failures. Therefore **148/150 questions qualified**. The original seed-42 split contained 75/37/38 questions; qualified split sizes were **74 train / 36 validation / 38 test**. The trust estimator was fit on the qualified train rows, sigmoid calibration on validation rows only, and all reported system metrics use the 38-row test split only.

### Per-model test results (n=38)

| Model | Raw accuracy | Raw ECE | Raw Brier | Calibrated accuracy | Calibrated ECE | Calibrated Brier |
|---|---:|---:|---:|---:|---:|---:|
| Groq GPT-OSS 20B | 92.11% | 0.0829 | 0.0848 | 97.37% | 0.0496 | 0.0302 |
| Groq GPT-OSS 120B | 97.37% | 0.0503 | 0.0226 | 92.11% | 0.0422 | 0.0292 |

### Four-system comparison (test n=38)

| System | Accuracy | Selective accuracy | Coverage | ECE | Brier | AURC |
|---|---:|---:|---:|---:|---:|---:|
| Best single, raw confidence (GPT-OSS 120B) | 97.37% | 97.37% | 100.00% | 0.0503 | 0.0226 | 0.0014 |
| Best single, calibrated confidence (GPT-OSS 120B) | 92.11% | 100.00% | 92.11% | 0.0422 | 0.0292 | 0.0014 |
| Majority/agreement baseline | 97.37% | 97.37% | 100.00% | 0.0263 | 0.0263 | 0.0263 |
| TrustMeter with reject option | 97.37% | 97.37% | 100.00% | 0.0042 | 0.0256 | 0.0107 |

The exact 101-point risk-coverage curves for all four systems are saved in `backend/models/metrics.json`. Each system row stores `total: 38`; `evaluation.test_set_size` is also 38.

The hypothesis is **not supported** by the predeclared AURC test on this split: TrustMeter AURC is 0.0107, greater than both best-single baselines at 0.0014. TrustMeter has lower ECE than the other systems, but the stated criterion is not met. This is a small test set, so the measured difference is uncertain and should not be generalized.

### Live smoke checks

These five previously recorded questions were asked in a prior smoke check after training and are not part of the ARC evaluation. Gemini was disabled; both active Groq models returned usable answers for all five. They were not re-queried during the cache-only verification in this run, because that verification made no provider calls.

| Question | GPT-OSS 20B | GPT-OSS 120B | Trust score | Verdict |
|---|---|---|---:|---|
| Chemical symbol of gold | Au | Au | 0.9701 | TRUSTED |
| Red Planet | Mars | Mars | 0.9701 | TRUSTED |
| 17 sheep, all but 9 die | 9 | 9 | 0.9701 | TRUSTED |
| Is a hot dog a sandwich? | Yes | Yes, generally; wording differed | 0.3311 | DISPUTED |
| Capital of nonexistent Blorvia | No capital; nonexistent | Fictional; no real capital | 0.3405 | DISPUTED |

The last two verdicts reflect exact normalized string matching, not a substantive factual disagreement: both providers gave compatible answers, but their phrasing differed. This demonstrates why live verdicts are estimates and why semantic answer grouping remains a limitation. The hot-dog disagreement also confirms that an explicit dissent is returned as `DISPUTED` even when the trust score is below threshold; agreeing answers below threshold remain `ABSTAIN`.

## Limitations

- The experiment uses only 150 questions from ARC-Challenge's multiple-choice test split; it does not represent broad factual QA.
- Hosted free-tier models may be rate-limited, changed, or retired. Their responses and self-reported confidence can vary.
- String-match grouping misses semantic equivalence and can group answers incorrectly.
- The two Groq GPT-OSS models are from the same model family, so their errors may be correlated; agreement between them is not independent evidence.
- The test metrics use only 38 qualified examples. Two questions were excluded because one cached Groq response each was not a valid ARC choice; the cached responses were preserved and not re-called.
- The trust model's historical accuracies are ARC-specific. Live verdicts on other topics are more uncertain.
- Calibration and abstention do not prove correctness or guarantee safety.

## Conclusion

**The hypothesis is not supported on the measured split.** TrustMeter did not achieve a lower test AURC than either best-single confidence baseline. The result is limited to these 38 test questions, two same-family models, and this cached run; it is not evidence that the hypothesis is false in every setting.
