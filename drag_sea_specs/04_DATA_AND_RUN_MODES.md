# 04 — Data and Run Modes Spec

## 1. Data folder reality

User data folder screenshot shows something like:

```text
data/flashrag_.../
├── 2wiki/
├── 2wikimultihopqa/
├── nq/
└── strategyqa/
```

DRAG README expects a project-level dataset folder with dataset names such as:

```text
dataset/
├── 2wiki
├── HotpotQA
├── NQ
├── PopQA
├── StrategyQA
└── TriviaQA
```

Implementation must not hard-code uppercase names. It should resolve aliases.

## 2. Dataset resolver

File: `scripts/run_drag_sea.py` or `model/data_utils.py`.

Function:

```python
def resolve_dataset_folder(data_dir: str, dataset_name: str) -> tuple[str, str]:
    """
    Return canonical dataset_name for config and actual folder path.
    """
```

Rules:

1. If exact folder exists, use it.
2. Else try case-insensitive match.
3. Else try alias list.
4. Else raise clear error listing available folders.

Aliases:

```python
ALIASES = {
    "2wiki": ["2wiki", "2wikimultihopqa", "2WikiMultihopQA", "2Wiki"],
    "nq": ["nq", "NQ", "NaturalQuestions"],
    "strategyqa": ["strategyqa", "StrategyQA"],
    "hotpotqa": ["hotpotqa", "HotpotQA"],
    "popqa": ["popqa", "PopQA"],
    "triviaqa": ["triviaqa", "TriviaQA"],
}
```

## 3. Dataset item normalization

FlashRAG dataset item usually has fields like:

```python
item.question
item.golden_answers
item.update_output(key, value)
```

But custom JSON/JSONL may differ. Add helper:

```python
def get_question(item) -> str:
    if hasattr(item, "question"):
        return item.question
    if isinstance(item, dict):
        for key in ["question", "query", "input", "prompt"]:
            if key in item:
                return item[key]
    raise ValueError("Cannot find question field")
```

```python
def get_gold_answers(item) -> list[str]:
    for key in ["golden_answers", "answers", "answer", "gold", "label"]:
        ...
```

## 4. Run modes

### 4.1 `question`

Input is direct question string. No dataset required.

```bash
python scripts/run_drag_sea.py --run_mode question --question "..."
```

Output:

```text
output/drag_sea/question/<timestamp>_episode.json
```

### 4.2 `single`

Run one sample from dataset.

```bash
python scripts/run_drag_sea.py --run_mode single --dataset_name 2wiki --sample_id 12
```

Should print:

```text
Question
Gold answers if available
Final answer
Stop reason
Query pool
Memory ops count
Output path
```

### 4.3 `batch`

Run multiple samples.

```bash
python scripts/run_drag_sea.py --run_mode batch --sample_num 100 --start_idx 0
```

Options:

```text
--sample_num N
--start_idx I
--end_idx J
--random_sample true/false
--seed 2024
--resume true/false
```

Batch output:

```text
output/drag_sea/<dataset>/<run_id>/episodes.jsonl
output/drag_sea/<dataset>/<run_id>/predictions.jsonl
output/drag_sea/<dataset>/<run_id>/metrics.json
output/drag_sea/<dataset>/<run_id>/memory_ops.jsonl
```

## 5. Resume logic

If `--resume true`, skip sample_id already present in `episodes.jsonl`.

Use stable key:

```python
sample_key = f"{dataset_name}:{sample_id}"
```

## 6. CLI arguments

Minimum:

```python
parser.add_argument("--run_mode", choices=["question", "single", "batch"], required=True)
parser.add_argument("--question", type=str)
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--data_dir", type=str, default="dataset/")
parser.add_argument("--sample_id", type=int, default=0)
parser.add_argument("--sample_num", type=int, default=None)
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=None)
parser.add_argument("--openai_model", type=str, default=None)
parser.add_argument("--memory_path", type=str, default="output/drag_sea/memory.sqlite")
parser.add_argument("--max_query_debate_rounds", type=int, default=3)
parser.add_argument("--max_answer_debate_rounds", type=int, default=3)
parser.add_argument("--adaptive_stopping", action="store_true")
parser.add_argument("--no_memory", action="store_true")
parser.add_argument("--no_dtcls", action="store_true")
parser.add_argument("--no_crds", action="store_true")
parser.add_argument("--save_prompts", action="store_true")
```
