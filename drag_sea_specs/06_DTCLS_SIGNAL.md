# 06 — DTCLS Signal Spec

## 1. Definition

DTCLS = **Debate-Trajectory Contrastive Lesson Signal**.

It extracts reusable lessons from the contrast between:

- Proponent vs Challenger arguments
- old query vs refined query
- retrieved docs before vs after query operation
- answer before vs after response debate
- successful vs failed outcome

## 2. File

```text
model/dtcls.py
```

## 3. Class interface

```python
class DTCLSExtractor:
    def __init__(self, generator: OpenAIGenerator, config: dict): ...

    def extract(self, episode: dict) -> dict:
        """
        Return:
        {
          "lessons": list[dict],
          "memory_ops": list[dict],
          "raw_output": dict,
        }
        """
```

## 4. Outcome labeling

Function:

```python
def infer_success_label(episode: dict) -> str:
    ...
```

Use priority:

1. If gold answers available and metric computed:
   - EM/F1 exact enough -> SUCCESS
   - wrong -> FAIL
2. Else use proxy:
   - evidence_support_score >= 0.75 and judge_confidence >= 0.75 -> SUCCESS
   - drift_risk_score >= 0.75 or evidence_support_score < 0.3 -> FAIL
   - otherwise UNKNOWN/PARTIAL

## 5. Contrast construction

Build list of contrast records.

### 5.1 Query contrast

For each query debate round:

```python
contrast = {
    "type": "query_round",
    "round_idx": round_idx,
    "proponent_claim": proponent_argument,
    "challenger_claim": challenger_argument,
    "judge_decision": judge_decision,
    "operation": operation,
    "query_pool_before": [...],
    "query_pool_after": [...],
    "docs_before_summary": [...],
    "docs_after_summary": [...],
    "outcome": success_label,
}
```

### 5.2 Answer contrast

For each answer round:

```python
contrast = {
    "type": "answer_round",
    "round_idx": round_idx,
    "proponent_answer": proponent_answer,
    "challenger_answer": challenger_answer,
    "judge_answer": judge_answer,
    "agreement_score": agreement_score,
    "evidence_support_score": evidence_support_score,
    "drift_risk_score": drift_risk_score,
    "stop_decision": stop_decision,
    "outcome": success_label,
}
```

## 6. Lesson schema

File: `model/schemas.py`

```python
DTCLS_LESSON_SCHEMA = {
  "type": "object",
  "properties": {
    "lessons": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "lesson_type": {"type": "string"},
          "target_role": {"type": "string"},
          "trigger_condition": {"type": "string"},
          "recommended_action": {"type": "string"},
          "entity_or_topic": {"type": ["string", "null"]},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1},
          "evidence": {"type": "string"},
          "failure_mode": {"type": ["string", "null"]},
          "expected_benefit": {"type": "string"}
        },
        "required": [
          "lesson_type", "target_role", "trigger_condition",
          "recommended_action", "entity_or_topic", "confidence",
          "evidence", "failure_mode", "expected_benefit"
        ],
        "additionalProperties": False
      }
    }
  },
  "required": ["lessons"],
  "additionalProperties": False
}
```

## 7. Prompt

System instruction:

```text
You extract reusable, role-specific lessons from a DRAG+SEA debate episode.
Do not summarize the whole episode. Extract only lessons that can improve future retrieval debate, response debate, judge decisions, or adaptive stopping.
Each lesson must have a clear trigger condition and recommended action.
Avoid generic lessons like "be careful" or "retrieve better".
Return only JSON matching the schema.
```

User prompt:

```text
Question: {question}
Dataset: {dataset_name}
Gold answers: {gold_answers}
Final answer: {final_answer}
Outcome label: {success_label}

Relevant memory used this episode:
{relevant_lessons}

Query-stage contrasts:
{query_contrasts}

Answer-stage contrasts:
{answer_contrasts}

Task:
Extract 0-5 reusable lessons.
Focus on role-specific debate tactics and retrieval/generation failure patterns.
```

## 8. Lesson validation

Reject lesson if:

- `confidence < 0.55`
- trigger/action too generic
- target_role invalid
- duplicate of existing active lesson with lower/equal confidence
- no evidence

Generic trigger examples to reject:

```text
"when answering questions"
"when retrieval is bad"
"when the model is unsure"
```

Good trigger examples:

```text
"question contains an ambiguous title shared by film and band entities"
"query asks for current role/status but lacks a temporal qualifier"
"retrieved docs contain exact answer in one doc and unrelated background in others"
```

## 9. Convert lessons to memory ops

Each accepted lesson becomes:

```python
{
    "op_type": "ADD",
    "memory_id": None,
    "payload": lesson,
    "reason": "DTCLS extracted from episode contrast"
}
```

If duplicate exists:

```python
{
    "op_type": "UPDATE",
    "memory_id": existing_id,
    "payload": {"confidence": new_confidence, "last_evidence": evidence},
    "reason": "DTCLS reinforced existing lesson"
}
```

## 10. Tests

`tests/test_dtcls.py`:

- Build fake successful episode where Challenger query expansion wins -> expect query_expansion lesson.
- Build fake failed episode where excessive expansion causes drift -> expect early_stop or drift lesson.
- Reject generic lesson.
- Deduplicate against memory.
