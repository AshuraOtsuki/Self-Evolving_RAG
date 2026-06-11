# 05 — Memory Bank Spec

## 1. Goal

Memory Bank stores reusable lessons and retrieval snapshots across episodes.

It should answer:

- What lessons are relevant to this new query?
- Which old facts/lessons are outdated?
- Which debate tactics repeatedly help or hurt?

## 2. Storage choice

Use SQLite for v1.

File default:

```text
output/drag_sea/memory.sqlite
```

Advantages:

- easy to inspect
- no external service
- durable across runs
- supports SQL filtering by topic/status/time

Optional semantic index:

- `MemoryBank` remains the canonical SQLite implementation.
- `Mem0MemoryBank` can be selected as an optional adapter when semantic lesson retrieval is needed.
- The adapter writes every lesson, episode, retrieval snapshot, and memory operation to SQLite, then indexes active lessons into mem0 with `source=drag_sea_lesson` metadata.
- If mem0 is unavailable, retrieval falls back to the SQLite hybrid scorer so experiments remain runnable.

## 3. Tables

### 3.1 `memory_entries`

```sql
CREATE TABLE IF NOT EXISTS memory_entries (
    memory_id TEXT PRIMARY KEY,
    lesson_type TEXT NOT NULL,
    trigger_condition TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    target_role TEXT NOT NULL,
    entity_or_topic TEXT,
    dataset_name TEXT,
    source_episode_id TEXT,
    confidence REAL NOT NULL DEFAULT 0.5,
    usefulness_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    expires_at TEXT,
    metadata_json TEXT
);
```

Allowed `lesson_type`:

```text
query_refinement
query_expansion
query_optimization
early_stop
judge_correction
temporal_retrieval
entity_disambiguation
source_conflict
retrieval_failure
answer_grounding
adaptive_stopping
```

Allowed `target_role`:

```text
query_proponent
query_challenger
query_judge
answer_proponent
answer_challenger
answer_judge
adaptive_controller
all
```

Allowed `status`:

```text
active
outdated
deprecated
deleted
candidate
```

### 3.2 `episode_logs`

```sql
CREATE TABLE IF NOT EXISTS episode_logs (
    episode_id TEXT PRIMARY KEY,
    dataset_name TEXT,
    sample_id TEXT,
    question TEXT NOT NULL,
    final_answer TEXT,
    gold_answers_json TEXT,
    success_label TEXT,
    stop_reason TEXT,
    created_at TEXT NOT NULL,
    episode_json TEXT NOT NULL
);
```

Allowed `success_label`:

```text
SUCCESS
FAIL
PARTIAL
UNKNOWN
```

### 3.3 `retrieval_snapshots`

```sql
CREATE TABLE IF NOT EXISTS retrieval_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    entity_or_topic TEXT,
    question TEXT NOT NULL,
    query_pool_json TEXT NOT NULL,
    docs_json TEXT NOT NULL,
    doc_fingerprints_json TEXT NOT NULL,
    extracted_facts_json TEXT,
    answer TEXT,
    created_at TEXT NOT NULL
);
```

### 3.4 `memory_operations`

```sql
CREATE TABLE IF NOT EXISTS memory_operations (
    op_id TEXT PRIMARY KEY,
    episode_id TEXT,
    op_type TEXT NOT NULL,
    memory_id TEXT,
    payload_json TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);
```

Allowed `op_type`:

```text
ADD
UPDATE
DELETE
DOWNWEIGHT
UPWEIGHT
MARK_OUTDATED
TEMPORAL_WARNING
NOOP
```

## 4. Python interface

```python
class MemoryBank:
    def __init__(self, path: str): ...
    def init_db(self) -> None: ...

    def add_memory(self, entry: dict) -> str: ...
    def update_memory(self, memory_id: str, patch: dict) -> None: ...
    def mark_outdated(self, memory_id: str, reason: str) -> None: ...
    def delete_memory(self, memory_id: str, soft: bool = True) -> None: ...

    def retrieve_lessons(
        self,
        question: str,
        entity_or_topic: str | None = None,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> list[dict]: ...

    def save_episode(self, episode: dict) -> None: ...
    def save_retrieval_snapshot(self, snapshot: dict) -> str: ...
    def get_recent_snapshots(self, entity_or_topic: str, limit: int = 5) -> list[dict]: ...
    def apply_memory_ops(self, ops: list[dict], episode_id: str) -> list[dict]: ...
```

## 5. Retrieval scoring v1

No vector DB required in v1. Use a simple hybrid score:

```python
score = (
    0.35 * keyword_overlap(question, trigger_condition + recommended_action)
    + 0.25 * topic_match(entity_or_topic, entry.entity_or_topic)
    + 0.20 * confidence
    + 0.10 * recency_score(last_updated)
    + 0.10 * usefulness_score(usefulness_count, failure_count)
)
```

Filter:

```python
status == "active"
score >= min_score
```

## 6. Memory injection format

Do not dump full DB into prompts. Format top-k lessons compactly:

```text
Relevant past lessons:
[1] Type: entity_disambiguation | Target: query_challenger | Trigger: ambiguous entity names | Action: add entity type or context to query | Confidence: 0.81
[2] Type: early_stop | Target: query_judge | Trigger: exact answer appears in high-quality docs with no conflict | Action: stop retrieval debate | Confidence: 0.75
```

## 7. Memory update rules

### ADD

Add if:

- lesson confidence >= 0.6
- lesson not duplicate of existing active memory
- lesson has clear trigger + action

### UPDATE

Update if:

- same trigger and target role exists
- new confidence higher
- or CRDS says temporal condition changed

### DOWNWEIGHT

Downweight if:

- lesson used in current episode but final answer failed
- or CRDS detects conflict but not enough to delete

### DELETE / MARK_OUTDATED

Use soft delete first:

```python
status = "outdated"
```

Hard delete only behind `--hard_delete_memory`.

## 8. Tests

`tests/test_memory_bank.py` should test:

- DB initialization.
- Add/retrieve memory.
- Duplicate detection.
- Apply UPDATE op.
- Mark outdated excludes entry from retrieval.
- Save episode and snapshot.
