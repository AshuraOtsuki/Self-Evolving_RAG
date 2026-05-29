import json
from pathlib import Path
from datasets import load_dataset

repo = "RUC-NLPIR/FlashRAG_datasets"
targets = ["nq", "2wikimultihopqa", "strategyqa"]
splits = ["train", "dev", "test"]
n = 30

out_root = Path("data/flashrag_tiny")
out_root.mkdir(parents=True, exist_ok=True)

for name in targets:
    ds_dir = out_root / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        out_file = ds_dir / f"{split}.jsonl"
        try:
            stream = load_dataset(repo, name, split=split, streaming=True)
            with out_file.open("w", encoding="utf-8") as f:
                for row in stream.take(n):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"OK  {name}/{split}.jsonl")
        except Exception as e:
            print(f"SKIP {name}/{split}: {e}")