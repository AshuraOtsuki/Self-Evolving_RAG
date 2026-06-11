# 08 — Adaptive Response Debate Spec

## 1. Goal

DRAG baseline uses fixed answer debate rounds. DRAG+SEA should stop early when enough reasoning has been done, or when debate starts causing problem drift.

## 2. File

```text
model/adaptive_stopping.py
```

## 3. Class interface

```python
class AdaptiveStoppingController:
    def __init__(self, config: dict): ...

    def score_round(
        self,
        question: str,
        query_pool: dict,
        answer_rounds: list[dict],
        current_proponent_answer: str,
        current_challenger_answer: str,
        current_judge_answer: str,
    ) -> dict: ...

    def should_stop(self, metrics: dict, round_idx: int) -> tuple[bool, str]: ...
```

## 4. Metrics

### 4.1 Agreement score

Measures whether Proponent, Challenger, Judge converge.

```python
agreement_score = max_pairwise_answer_similarity([
  extract_answer(proponent_answer),
  extract_answer(challenger_answer),
  extract_answer(judge_answer),
])
```

v1 similarity:

- exact normalized match -> 1.0
- token F1
- yes/no exact for StrategyQA

### 4.2 Evidence support score

Measures whether judge answer is supported by retrieved docs.

v1 heuristic:

```python
answer = extract_answer(judge_answer)
docs_text = concatenate_retrieved_docs(query_pool)
if answer exact substring in docs_text: 1.0
elif token_f1(answer, docs_text) high: 0.7
else: 0.2
```

Optional LLM verifier with structured output:

```json
{
  "support_score": 0.82,
  "supporting_doc_ids": ["..."],
  "contradiction_found": false,
  "reason": "..."
}
```

Use LLM verifier only if `adaptive_stopping.use_llm_evidence_verifier=true` to control cost.

### 4.3 Answer stability score

If judge answer does not change across rounds:

```python
if round_idx == 0: stability = 0.5
else: stability = similarity(current_judge_answer, previous_judge_answer)
```

### 4.4 Drift risk score

Detect if debate is moving away from original question.

Heuristics:

- New named entities not present in question or retrieved docs.
- Answer changes but evidence support decreases.
- Challenger introduces unsupported speculation.
- Query topic and answer topic diverge.

v1:

```python
drift_risk = 0.0
if evidence_support_score < previous_evidence_support_score - 0.2:
    drift_risk += 0.3
if answer_stability_score < 0.4 and round_idx >= 1:
    drift_risk += 0.2
if unsupported_entity_count > 0:
    drift_risk += 0.3
if agreement_score < 0.3 and round_idx >= 2:
    drift_risk += 0.2
```

Clamp to [0, 1].

## 5. Stopping rule

Config:

```yaml
adaptive_stopping:
  min_rounds: 1
  max_rounds: 3
  agreement_threshold: 0.85
  evidence_threshold: 0.75
  stability_threshold: 0.85
  drift_risk_threshold: 0.65
```

Logic:

```python
if round_idx + 1 < min_rounds:
    return False, "CONTINUE_MIN_ROUNDS"

if drift_risk_score >= drift_risk_threshold:
    return True, "STOP_DRIFT_RISK"

if agreement_score >= agreement_threshold and evidence_support_score >= evidence_threshold:
    return True, "STOP_CONFIDENT_AGREEMENT"

if evidence_support_score >= evidence_threshold and answer_stability_score >= stability_threshold:
    return True, "STOP_STABLE_GROUNDED"

if round_idx + 1 >= max_rounds:
    return True, "STOP_MAX_ROUNDS"

return False, "CONTINUE"
```

## 6. Integration in answer stage

Current baseline:

```python
for round in range(self.max_answer_debate_rounds):
    ...
```

DRAG+SEA:

```python
answer_rounds = []
for round_idx in range(self.max_answer_debate_rounds):
    proponent = call_answer_proponent(...)
    challenger = call_answer_challenger(...)
    judge = call_answer_judge(...)

    metrics = self.adaptive_controller.score_round(...)
    stop, reason = self.adaptive_controller.should_stop(metrics, round_idx)

    answer_rounds.append({...metrics..., "stop_decision": reason})

    if stop:
        break
```

## 7. Output

Always log stop reason:

```python
item.update_output("AnswerStage_StopReason", stop_reason)
item.update_output("AnswerStage_StopMetrics", stop_metrics)
```

Episode:

```python
"answer_stage": {
    "rounds": answer_rounds,
    "final_answer": final_answer,
    "stop_reason": stop_reason,
    "stop_metrics": stop_metrics,
}
```

## 8. Tests

`tests/test_adaptive_stopping.py`:

- High agreement + evidence -> stop.
- Low round below min -> continue.
- High drift risk -> stop drift.
- Max rounds -> stop max.
- Answer stable but agreement moderate -> stop stable grounded.
