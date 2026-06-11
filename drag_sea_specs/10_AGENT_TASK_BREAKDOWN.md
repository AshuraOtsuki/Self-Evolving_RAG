# 10 — Agent Task Breakdown

This file is written as direct implementation tasks for coding agents.

## Agent A — Repo & CLI Integration

### Goal

Create runnable script without breaking baseline.

### Tasks

1. Add `scripts/run_drag_sea.py`.
2. Add dataset alias resolver.
3. Load FlashRAG config from `config/base_config.yaml` plus CLI overrides.
4. Support `question`, `single`, `batch` modes.
5. Save output directory and JSONL files.
6. Add `DRAG_SEA` method optionally to `main.py` or keep script-only.

### Acceptance

```bash
python scripts/run_drag_sea.py --run_mode question --question "test" --dry_run true
```

prints planned config and exits.

## Agent B — OpenAI Generator Adapter

### Goal

Implement OpenAI-backed generator compatible with `.generate()`.

### Tasks

1. Create `model/openai_generator.py`.
2. Implement `OpenAIGeneratorConfig`.
3. Implement `.generate()`.
4. Implement `.generate_json()` with strict schema.
5. Add retry/backoff.
6. Add unit tests with mocked OpenAI client.

### Acceptance

- Existing DRAG-style prompt call works:

```python
out = generator.generate("Question: hi")
assert isinstance(out, list)
assert isinstance(out[0], str)
```

## Agent C — Memory Bank

### Goal

Persistent SQLite memory.

### Tasks

1. Create `model/memory_bank.py`.
2. Create DB tables.
3. Implement CRUD.
4. Implement `retrieve_lessons` hybrid lexical scoring.
5. Implement `apply_memory_ops`.
6. Add `scripts/inspect_memory.py`.
7. Unit tests.

### Acceptance

```bash
python scripts/inspect_memory.py --memory_path output/drag_sea/memory.sqlite --list active
```

prints table even if empty.

## Agent D — DRAGSEAPipeline

### Goal

Implement pipeline orchestration.

### Tasks

1. Create `model/drag_sea.py`.
2. Reuse baseline DRAG structure.
3. Add memory retrieval before query debate.
4. Add structured query debate outputs.
5. Add adaptive answer debate.
6. Save episode dict.
7. Trigger DTCLS and CRDS post episode.

### Acceptance

One sample produces complete episode dict with:

```python
episode["query_stage"]
episode["answer_stage"]
episode["dtcls"]
episode["crds"]
episode["usage"]
```

## Agent E — Prompt & Schema Engineer

### Goal

Create stable prompts and JSON schemas.

### Tasks

1. Create `prompts/drag_sea_prompts.py`.
2. Create `model/schemas.py`.
3. Define schemas for:
   - query proponent
   - query challenger
   - query judge
   - answer proponent optional structured
   - answer challenger optional structured
   - answer judge
   - DTCLS lesson extraction
   - CRDS entity extraction
   - CRDS drift decision
   - evidence verifier
4. Make all parsed outputs strict JSON.

### Acceptance

All schemas pass `jsonschema.Draft202012Validator.check_schema(schema)`.

## Agent F — DTCLS

### Goal

Extract lessons after episode.

### Tasks

1. Create `model/dtcls.py`.
2. Implement contrast construction.
3. Implement success labeling.
4. Call OpenAI structured output.
5. Validate and deduplicate lessons.
6. Convert lessons to memory ops.
7. Unit tests.

### Acceptance

Fake episode with Challenger successful query expansion generates an ADD op.

## Agent G — CRDS

### Goal

Cross-session drift detection.

### Tasks

1. Create `model/crds.py`.
2. Implement snapshot builder.
3. Implement entity/topic extraction.
4. Implement drift score.
5. Implement memory ops.
6. Unit tests with old/current snapshots.

### Acceptance

Old answer A, current answer B for same current-status topic creates temporal warning + mark outdated op.

## Agent H — Adaptive Stopping

### Goal

Reduce fixed answer rounds.

### Tasks

1. Create `model/adaptive_stopping.py`.
2. Implement answer extraction.
3. Implement agreement/evidence/stability/drift scores.
4. Implement stopping rule.
5. Integrate in `DRAGSEAPipeline.answer_stage_debate`.
6. Unit tests.

### Acceptance

When Proponent and Challenger agree and answer appears in docs, stops after min rounds.

## Agent I — Evaluation & Reports

### Goal

Make results analyzable.

### Tasks

1. Implement metrics aggregation.
2. Implement run summary.
3. Implement `scripts/export_episode_logs.py`.
4. Add `metrics.json`.
5. Add cost/usage log if usage available.

### Acceptance

Batch run produces:

```text
episodes.jsonl
predictions.jsonl
metrics.json
memory_ops.jsonl
```
