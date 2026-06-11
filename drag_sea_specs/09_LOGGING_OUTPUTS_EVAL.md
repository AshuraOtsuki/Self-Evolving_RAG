# 09 — Logging, Outputs, and Evaluation Spec

## 1. Output directory

Default:

```text
output/drag_sea/<dataset_name>/<run_id>/
```

For direct question mode:

```text
output/drag_sea/question/<run_id>/
```

## 2. Files

### 2.1 `episodes.jsonl`

One JSON object per episode. Contains full transcript if `save_prompts=true`.

### 2.2 `predictions.jsonl`

Lightweight:

```json
{
  "sample_id": 0,
  "question": "...",
  "pred": "...",
  "golden_answers": ["..."],
  "stop_reason": "STOP_CONFIDENT_AGREEMENT",
  "query_count": 2,
  "answer_rounds": 1,
  "llm_calls": 7,
  "retriever_calls": 2
}
```

### 2.3 `metrics.json`

```json
{
  "em": 0.0,
  "f1": 0.0,
  "accuracy": 0.0,
  "avg_llm_calls": 0.0,
  "avg_retriever_calls": 0.0,
  "avg_answer_rounds": 0.0,
  "stop_reason_counts": {},
  "memory_ops_counts": {},
  "dtcls_lesson_count": 0,
  "crds_drift_count": 0
}
```

### 2.4 `memory_ops.jsonl`

```json
{
  "episode_id": "...",
  "op_type": "ADD",
  "memory_id": "...",
  "payload": {...},
  "reason": "DTCLS extracted from episode contrast"
}
```

### 2.5 `cost_usage.jsonl`

If usage data available from OpenAI:

```json
{
  "episode_id": "...",
  "stage": "query_proponent",
  "model": "...",
  "input_tokens": 123,
  "output_tokens": 45,
  "created_at": "..."
}
```

## 3. Episode logging requirements

Every LLM call should be attributable to:

```text
stage
round_idx
agent_name
prompt_hash
raw_output
parsed_output if JSON
```

Recommended agent names:

```text
query_proponent
query_challenger
query_judge
answer_proponent
answer_challenger
answer_judge
dtcls_extractor
crds_detector
evidence_verifier
entity_extractor
```

## 4. Evaluation

### 4.1 Standard QA metrics

Implement or reuse FlashRAG metrics:

- EM
- F1
- accuracy for StrategyQA yes/no

### 4.2 DRAG+SEA efficiency metrics

```python
avg_llm_calls
avg_retriever_calls
avg_query_debate_rounds
avg_answer_debate_rounds
avg_latency_seconds
```

### 4.3 Memory metrics

```python
lessons_added
lessons_updated
lessons_marked_outdated
lessons_retrieved_per_episode
memory_hit_rate
memory_usefulness_proxy
```

Memory usefulness proxy:

- if relevant lesson used and episode success -> +1 usefulness
- if relevant lesson used and episode fail -> +1 failure

### 4.4 Adaptive stopping metrics

```python
stop_reason_counts
avg_rounds_saved = max_answer_rounds - actual_answer_rounds
estimated_llm_calls_saved
```

## 5. Debug report

Add script:

```bash
python scripts/export_episode_logs.py --run_dir output/drag_sea/2wiki/<run_id> --sample_id 0
```

Output markdown:

```text
episode_<sample_id>_report.md
```

Report sections:

1. Question / gold / pred
2. Retrieved lessons
3. Query debate table
4. Final query pool and docs
5. Answer debate table
6. Stop metrics
7. DTCLS lessons
8. CRDS memory ops

## 6. Failure handling

If a sample fails due to API or parsing:

- write to `errors.jsonl`
- continue batch if `--continue_on_error true`
- include traceback hash, not full secret/environment

```json
{
  "sample_id": 12,
  "question": "...",
  "stage": "answer_judge",
  "error_type": "StructuredOutputError",
  "message": "missing key winner",
  "created_at": "..."
}
```
