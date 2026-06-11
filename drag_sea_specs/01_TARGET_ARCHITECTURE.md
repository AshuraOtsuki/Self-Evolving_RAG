# 01 — Target Architecture

## 1. High-level pipeline

```text
Input question x
   |
   v
[Memory Lesson Retrieval]
   |
   v
[Retrieval Debate]
   |  logs query trajectory
   v
[Final Query Pool + Retrieved Docs]
   |
   v
[Response Debate + Adaptive Stopping]
   |  logs answer trajectory
   v
[Final Judge Answer]
   |
   v
[Post-Episode Self-Evolution]
   |-- DTCLS: extract debate lessons
   |-- CRDS: detect retrieval drift
   v
[Memory Bank ADD / UPDATE / DELETE / DOWNWEIGHT]
```

## 2. Main classes

### 2.1 `OpenAIGenerator`

File: `model/openai_generator.py`

Responsibility:

- Wrap OpenAI Responses API.
- Expose `.generate(prompts: list[str] | str) -> list[str]`.
- Support normal text mode and strict JSON schema mode.
- Retry transient failures.
- Log token usage/cost metadata if available.

### 2.2 `DRAGSEAPipeline`

File: `model/drag_sea.py`

Responsibility:

- Orchestrate the whole DRAG+SEA episode.
- Keep compatibility with DRAG style: `pipeline.run(dataset)`.
- Add `run_one(item)` for debug.
- Add `run_question(question)` for custom question.

Required public methods:

```python
class DRAGSEAPipeline(BasicPipeline):
    def run(self, dataset, do_eval: bool = True): ...
    def run_one(self, item, do_eval: bool = False) -> dict: ...
    def run_question(self, question: str) -> dict: ...
    def query_stage_debate(self, item, relevant_lessons: list[dict]) -> dict: ...
    def answer_stage_debate(self, item, query_pool: dict, relevant_lessons: list[dict]) -> dict: ...
    def post_episode_update(self, episode: dict) -> dict: ...
```

### 2.3 `MemoryBank`

File: `model/memory_bank.py`

Responsibility:

- SQLite storage.
- CRUD memory entries.
- Store episode logs and retrieval snapshots.
- Retrieve top-k lessons using lexical/embedding similarity.

### 2.4 `LessonRetriever`

File: `model/lesson_retriever.py`

Responsibility:

- Given question + optional topic/entity, retrieve relevant lessons.
- For v1, use hybrid score:
  - keyword overlap
  - entity/topic match
  - lesson confidence
  - recency
  - status active

### 2.5 `DTCLSExtractor`

File: `model/dtcls.py`

Responsibility:

- Build contrastive pairs from query/answer trajectory.
- Call OpenAI structured output to produce lessons.
- Validate lessons.

### 2.6 `CRDSDetector`

File: `model/crds.py`

Responsibility:

- Store retrieval snapshots.
- Compare old and new doc sets.
- Generate memory operations when drift is detected.

### 2.7 `AdaptiveStoppingController`

File: `model/adaptive_stopping.py`

Responsibility:

- After each response debate round, compute stop metrics.
- Decide `CONTINUE`, `STOP_CONFIDENT`, `STOP_DRIFT_RISK`, `STOP_MAX_ROUNDS`.

## 3. Data objects

### 3.1 Episode object

Every sample should produce an episode dict:

```python
episode = {
    "episode_id": str,
    "dataset_name": str | None,
    "sample_id": str | int | None,
    "question": str,
    "golden_answers": list[str] | None,
    "relevant_lessons": list[dict],
    "query_stage": {
        "rounds": list[dict],
        "final_query_pool": dict,
    },
    "answer_stage": {
        "rounds": list[dict],
        "final_answer": str,
        "stop_reason": str,
        "stop_metrics": dict,
    },
    "retrieval_snapshot": dict,
    "dtcls": {
        "lessons": list[dict],
        "raw_output": dict | None,
    },
    "crds": {
        "drift_score": float | None,
        "memory_ops": list[dict],
        "raw_output": dict | None,
    },
    "usage": {
        "llm_calls": int,
        "retriever_calls": int,
        "estimated_input_tokens": int | None,
        "estimated_output_tokens": int | None,
    },
}
```

### 3.2 Query round object

```python
query_round = {
    "round_idx": int,
    "query_pool_before": list[str],
    "retrieved_docs_before": list[dict],
    "proponent_argument": str,
    "challenger_argument": str,
    "judge_decision": "PROPONENT" | "CHALLENGER",
    "judge_reason": str | None,
    "operation": "KEEP" | "QUERY_OPTIMIZATION" | "QUERY_EXPANSION" | "STOP",
    "operation_payload": dict,
    "query_pool_after": list[str],
}
```

### 3.3 Answer round object

```python
answer_round = {
    "round_idx": int,
    "proponent_answer": str,
    "challenger_answer": str,
    "judge_answer": str,
    "judge_confidence": float,
    "agreement_score": float,
    "evidence_support_score": float,
    "answer_stability_score": float,
    "drift_risk_score": float,
    "stop_decision": str,
}
```

## 4. Design rule

All agent outputs that are parsed by code should use JSON schema. Only final answer may be plain text if matching benchmark format is needed.
