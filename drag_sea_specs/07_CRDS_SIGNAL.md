# 07 — CRDS Signal Spec

## 1. Definition

CRDS = **Cross-Session Retrieval Drift Signal**.

It compares retrieval snapshots across sessions for the same or related entity/topic. If docs/facts drift, it updates memory.

CRDS answers:

- Did the retrieved evidence for this topic change significantly?
- Are old memory entries now outdated?
- Should future queries require temporal qualifiers?

## 2. File

```text
model/crds.py
```

## 3. Class interface

```python
class CRDSDetector:
    def __init__(self, memory_bank: MemoryBank, generator: OpenAIGenerator, config: dict): ...

    def build_snapshot(self, episode: dict) -> dict: ...

    def detect(self, episode: dict, current_snapshot: dict) -> dict:
        """
        Return:
        {
          "entity_or_topic": str | None,
          "old_snapshots": list[dict],
          "drift_score": float,
          "drift_type": str,
          "memory_ops": list[dict],
          "raw_output": dict | None,
        }
        """
```

## 4. Entity/topic key extraction

v1 approach:

1. Ask OpenAI structured output to extract:
   - main entity
   - relation
   - temporal intent
   - topic key
2. Fallback heuristic: noun phrase from question.

Schema:

```json
{
  "entity_or_topic": "OpenAI CEO",
  "main_entity": "OpenAI",
  "relation": "CEO",
  "temporal_intent": "current",
  "requires_freshness": true,
  "confidence": 0.82
}
```

## 5. Snapshot structure

```python
snapshot = {
    "snapshot_id": str,
    "episode_id": str,
    "entity_or_topic": str | None,
    "question": str,
    "query_pool": list[str],
    "docs": [
        {
            "doc_id": str | None,
            "title": str,
            "text": str,
            "fingerprint": str,
            "rank": int,
            "query": str,
            "timestamp": str | None,
            "source": str | None,
        }
    ],
    "extracted_facts": [
        {
            "subject": str,
            "predicate": str,
            "object": str,
            "time_scope": str | None,
            "supporting_doc_fingerprint": str,
        }
    ],
    "answer": str,
    "created_at": str,
}
```

Fingerprint:

```python
fingerprint = sha256(normalize(title + "\n" + first_500_chars(text)))
```

## 6. Drift score

Compute:

```python
drift_score = (
    0.30 * doc_set_distance
    + 0.25 * fact_contradiction_score
    + 0.20 * answer_change_score
    + 0.15 * timestamp_gap_score
    + 0.10 * source_change_score
)
```

### 6.1 Doc set distance

```python
doc_overlap = len(current_fingerprints & old_fingerprints) / max(1, len(current_fingerprints | old_fingerprints))
doc_set_distance = 1 - doc_overlap
```

### 6.2 Fact contradiction score

v1:

- If extracted facts for same subject+predicate have different objects -> 1.0
- Else 0.0

Optional LLM check for nuanced contradiction.

### 6.3 Answer change score

Normalize final answer string.

```python
answer_change_score = 0 if normalized_old_answer == normalized_current_answer else 1
```

For long answers, use token F1 distance.

### 6.4 Timestamp gap score

If current topic requires freshness:

```python
days = current_created_at - old_created_at
score = min(days / 180, 1.0)
```

Else score lower:

```python
score = min(days / 365, 1.0) * 0.5
```

## 7. Drift type

Allowed:

```text
NO_DRIFT
DOC_SET_DRIFT
FACT_CONTRADICTION
ANSWER_CHANGE
TEMPORAL_STALENESS
RETRIEVAL_QUALITY_DROP
UNKNOWN
```

Rules:

- `drift_score < 0.25`: NO_DRIFT
- `fact_contradiction_score >= 0.8`: FACT_CONTRADICTION
- `answer_change_score == 1 and requires_freshness`: ANSWER_CHANGE
- `timestamp_gap_score >= 0.8 and requires_freshness`: TEMPORAL_STALENESS
- `doc_set_distance >= 0.7`: DOC_SET_DRIFT

## 8. Memory operations

### 8.1 Fact contradiction

```python
{
  "op_type": "MARK_OUTDATED",
  "memory_id": old_memory_id,
  "payload": {"status": "outdated"},
  "reason": "CRDS detected contradiction between old and new retrieved facts"
}
```

### 8.2 Temporal warning

```python
{
  "op_type": "ADD",
  "payload": {
    "lesson_type": "temporal_retrieval",
    "target_role": "query_challenger",
    "trigger_condition": "query asks about current or time-sensitive status for {entity_or_topic}",
    "recommended_action": "add explicit temporal qualifier and prefer fresh retrieved evidence",
    "entity_or_topic": entity_or_topic,
    "confidence": 0.75
  },
  "reason": "CRDS detected temporal drift"
}
```

### 8.3 Retrieval quality dropped

If old retrieval had strong evidence but current retrieval does not:

```python
{
  "op_type": "ADD",
  "payload": {
    "lesson_type": "retrieval_failure",
    "target_role": "query_challenger",
    "trigger_condition": "retrieval for this topic returns low-overlap or weak evidence documents",
    "recommended_action": "reformulate query using entity + relation + temporal qualifier",
    "entity_or_topic": entity_or_topic,
    "confidence": 0.65
  },
  "reason": "CRDS detected retrieval quality drop"
}
```

## 9. Tests

`tests/test_crds.py`:

- Same docs -> drift low.
- Different docs, same answer -> doc set drift medium.
- Same entity with different answer -> answer change high.
- Current query topic -> temporal warning op created.
- Outdated memory excluded after op.
