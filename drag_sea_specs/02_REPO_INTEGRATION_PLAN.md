# 02 — Repo Integration Plan

## 1. Do not edit baseline first

Tạo file mới thay vì sửa mạnh vào baseline:

```text
model/drag_sea.py
model/openai_generator.py
model/memory_bank.py
model/lesson_retriever.py
model/dtcls.py
model/crds.py
model/adaptive_stopping.py
model/schemas.py
prompts/drag_sea_prompts.py
scripts/run_drag_sea.py
config/drag_sea_config.yaml
```

Chỉ sửa nhỏ:

```text
model/__init__.py
```

để export class mới nếu cần.

Có thể sửa `config/init_config.py` để thêm choice `DRAG_SEA`, nhưng nên ưu tiên script riêng `scripts/run_drag_sea.py` để tránh phá baseline.

## 2. Existing DRAG entry points to reuse

Từ `model/drag.py`, reuse logic:

- `format_query_pool`
- `_format_reference`
- `maintain_query_pool`
- `find_most_similar_key`

Nhưng version mới nên parse structured JSON thay vì parse string bằng `"Query Optimization:"`.

## 3. Replace string parsing with structured action

Baseline opponent output format:

```text
Query Optimization: [Original Query] -> [New Query]
Query Expansion: [New Query]
```

DRAG+SEA structured action:

```json
{
  "role": "challenger",
  "argument": "...",
  "operation": "QUERY_OPTIMIZATION",
  "original_query": "...",
  "new_query": "...",
  "missing_information": ["..."],
  "confidence": 0.82
}
```

Judge output:

```json
{
  "winner": "CHALLENGER",
  "decision": "APPLY_OPERATION",
  "reason": "...",
  "operation": {
    "type": "QUERY_EXPANSION",
    "query": "..."
  }
}
```

## 4. Add method map option

Nếu muốn integrate vào `main.py`, thêm:

```python
from model.drag_sea import drag_sea

func_map = {
    ...,
    "DRAG_SEA": drag_sea,
}
```

Nhưng vì DRAG+SEA cần nhiều args OpenAI/memory, script riêng dễ hơn.

## 5. Script `scripts/run_drag_sea.py`

Responsibilities:

- Parse CLI.
- Load FlashRAG config.
- Resolve dataset aliases.
- Create retriever via `get_retriever(cfg)`.
- Create `OpenAIGenerator`.
- Create `MemoryBank`.
- Create `DRAGSEAPipeline`.
- Run mode:
  - `single`
  - `batch`
  - `question`
- Save outputs.

Pseudo:

```python
def main():
    args = parse_args()
    cfg = build_flashrag_config(args)
    dataset = load_dataset_if_needed(args, cfg)
    generator = OpenAIGenerator(...)
    memory = MemoryBank(args.memory_path)
    pipeline = DRAGSEAPipeline(cfg, generator=generator, retriever=get_retriever(cfg), memory_bank=memory, ...)

    if args.run_mode == "question":
        result = pipeline.run_question(args.question)
    elif args.run_mode == "single":
        item = dataset[args.sample_id]
        result = pipeline.run_one(item)
    elif args.run_mode == "batch":
        result = pipeline.run(dataset_slice)
```

## 6. Config fields to add

`config/drag_sea_config.yaml` should contain:

```yaml
openai:
  model: "gpt-4.1-mini"
  reasoning_effort: null
  temperature: 0.0
  max_output_tokens: 512
  timeout_seconds: 60
  max_retries: 3
  structured_output: true

memory:
  enabled: true
  path: "./output/drag_sea/memory.sqlite"
  top_k_lessons: 5
  min_lesson_score: 0.15
  decay_half_life_days: 60

adaptive_stopping:
  enabled: true
  min_rounds: 1
  max_rounds: 3
  agreement_threshold: 0.85
  evidence_threshold: 0.75
  stability_threshold: 0.85
  drift_risk_threshold: 0.65

signals:
  dtcls_enabled: true
  crds_enabled: true
  extract_lessons_after_each_episode: true
  apply_memory_ops: true

run:
  save_episode_jsonl: true
  save_prompts: true
  save_raw_llm_outputs: true
```

## 7. Dataset alias mapping

Ảnh user có folder như:

```text
2wiki
2wikimultihopqa
nq
strategyqa
```

FlashRAG/DRAG gốc dùng tên có thể là:

```text
2wiki
NQ
StrategyQA
HotpotQA
PopQA
TriviaQA
```

Cần mapper:

```python
DATASET_ALIASES = {
    "2wiki": ["2wiki", "2Wiki", "2WikiMultihopQA", "2wikimultihopqa"],
    "nq": ["nq", "NQ", "NaturalQuestions"],
    "strategyqa": ["strategyqa", "StrategyQA"],
    "hotpotqa": ["hotpotqa", "HotpotQA"],
    "popqa": ["popqa", "PopQA"],
    "triviaqa": ["triviaqa", "TriviaQA"],
}
```

`resolve_dataset_name(name, data_dir)` should choose existing folder if possible.
