import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import dotenv

try:
    import yaml
except Exception:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.adaptive_stopping import normalize_answer, token_f1
from model.drag_sea import DRAGSEAPipeline
from model.memory_bank import MemoryBank
from model.openai_generator import OpenAIGenerator, OpenAIGeneratorConfig


ALIASES = {
    "2wiki": ["2wiki", "2wikimultihopqa", "2WikiMultihopQA", "2Wiki"],
    "nq": ["nq", "NQ", "NaturalQuestions"],
    "strategyqa": ["strategyqa", "StrategyQA"],
    "hotpotqa": ["hotpotqa", "HotpotQA"],
    "popqa": ["popqa", "PopQA"],
    "triviaqa": ["triviaqa", "TriviaQA"],
}


class MetadataRetriever:
    def __init__(self, items, topk=3):
        self.items = items or []
        self.topk = topk

    def search(self, query):
        scored = []
        query_terms = set(normalize_answer(query).split())
        for idx, item in enumerate(self.items):
            metadata = item.get("metadata") or {}
            facts = metadata.get("facts") or []
            text = " ".join([item.get("question", ""), metadata.get("term", ""), metadata.get("description", ""), *facts])
            terms = set(normalize_answer(text).split())
            score = len(query_terms & terms) / max(1, len(query_terms))
            scored.append((score, idx, item, text))
        scored.sort(key=lambda row: row[0], reverse=True)
        docs = []
        for score, idx, item, text in scored[: self.topk]:
            metadata = item.get("metadata") or {}
            facts = metadata.get("facts") or []
            title = metadata.get("term") or item.get("id") or f"item_{idx}"
            body = "\n".join([metadata.get("description", ""), *facts]).strip() or text
            docs.append(
                {
                    "id": item.get("id", str(idx)),
                    "title": title,
                    "text": body,
                    "contents": f"{title}\n{body}",
                    "score": score,
                    "source": "metadata_fallback",
                }
            )
        return docs


def parse_args():
    parser = argparse.ArgumentParser(description="Run DRAG+SEA with OpenAI and local FlashRAG-style JSONL data.")
    parser.add_argument("--run_mode", choices=["question", "single", "batch"], required=True)
    parser.add_argument("--question", type=str)
    parser.add_argument("--dataset_name", type=str, default="strategyqa")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data" / "flashrag_tiny"))
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--openai_model", type=str, default=None)
    parser.add_argument("--memory_path", type=str, default=str(REPO_ROOT / "output" / "drag_sea" / "memory.sqlite"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "config" / "drag_sea_config.yaml"))
    parser.add_argument("--max_query_debate_rounds", type=int, default=3)
    parser.add_argument("--max_answer_debate_rounds", type=int, default=3)
    parser.add_argument("--adaptive_stopping", action="store_true", default=True)
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument("--no_dtcls", action="store_true")
    parser.add_argument("--no_crds", action="store_true")
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_dataset_folder(data_dir: str, dataset_name: str) -> tuple[str, Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_path}")
    available = [path.name for path in data_path.iterdir() if path.is_dir()]
    if dataset_name in available:
        return canonical_name(dataset_name), data_path / dataset_name
    lower_map = {name.lower(): name for name in available}
    if dataset_name.lower() in lower_map:
        actual = lower_map[dataset_name.lower()]
        return canonical_name(actual), data_path / actual
    names_to_try = ALIASES.get(dataset_name.lower(), [dataset_name])
    for alias in names_to_try:
        if alias in available:
            return canonical_name(alias), data_path / alias
        if alias.lower() in lower_map:
            actual = lower_map[alias.lower()]
            return canonical_name(actual), data_path / actual
    raise FileNotFoundError(
        f"Dataset folder for '{dataset_name}' not found under {data_path}. Available: {', '.join(available)}"
    )


def canonical_name(name):
    key = name.lower()
    if key in {"strategyqa"}:
        return "StrategyQA"
    if key in {"nq", "naturalquestions"}:
        return "NQ"
    if key in {"2wiki", "2wikimultihopqa"}:
        return "2wiki"
    return name


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_config(path):
    defaults = {
        "openai": {
            "model": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_output_tokens": 512,
            "timeout_seconds": 60,
            "max_retries": 3,
        },
        "memory": {"enabled": True, "top_k_lessons": 5, "min_lesson_score": 0.15},
        "adaptive_stopping": {
            "enabled": True,
            "min_rounds": 1,
            "max_rounds": 3,
            "agreement_threshold": 0.85,
            "evidence_threshold": 0.75,
            "stability_threshold": 0.85,
            "drift_risk_threshold": 0.65,
        },
        "signals": {"dtcls_enabled": True, "crds_enabled": True, "apply_memory_ops": True},
        "run": {"save_prompts": False},
    }
    if yaml is None:
        return defaults
    with open(path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    for key, value in defaults.items():
        loaded.setdefault(key, value)
    return loaded


def build_output_dir(args, dataset_name):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) if args.output_dir else REPO_ROOT / "output" / "drag_sea"
    if args.run_mode == "question":
        return root / "question" / run_id
    return root / dataset_name / run_id


def make_generator(args, cfg):
    openai_cfg = cfg.get("openai", {})
    model = args.openai_model or os.getenv("OPENAI_MODEL") or openai_cfg.get("model", "gpt-4.1-mini")
    return OpenAIGenerator(
        OpenAIGeneratorConfig(
            model=model,
            temperature=float(openai_cfg.get("temperature", 0.0) or 0.0),
            max_output_tokens=int(openai_cfg.get("max_output_tokens", 512) or 512),
            timeout_seconds=int(openai_cfg.get("timeout_seconds", 60) or 60),
            max_retries=int(openai_cfg.get("max_retries", 3) or 3),
            reasoning_effort=openai_cfg.get("reasoning_effort"),
        )
    )


def write_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_from_episode(episode):
    return {
        "sample_id": episode.get("sample_id"),
        "question": episode.get("question"),
        "pred": episode.get("answer_stage", {}).get("final_answer"),
        "golden_answers": episode.get("golden_answers"),
        "stop_reason": episode.get("answer_stage", {}).get("stop_reason"),
        "query_count": len(episode.get("query_stage", {}).get("final_query_pool", {})),
        "answer_rounds": len(episode.get("answer_stage", {}).get("rounds", [])),
        "llm_calls": episode.get("usage", {}).get("llm_calls"),
        "retriever_calls": episode.get("usage", {}).get("retriever_calls"),
    }


def compute_metrics(episodes):
    total = len(episodes)
    em = 0.0
    f1 = 0.0
    acc = 0.0
    stop_counts = Counter()
    op_counts = Counter()
    for episode in episodes:
        pred = episode.get("answer_stage", {}).get("final_answer", "")
        golds = episode.get("golden_answers") or []
        if golds:
            em += max(1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0 for gold in golds)
            f1 += max(token_f1(pred, gold) for gold in golds)
            acc += max(1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0 for gold in golds)
        stop_counts[episode.get("answer_stage", {}).get("stop_reason", "UNKNOWN")] += 1
        for op in episode.get("memory_ops", []):
            op_counts[op.get("op_type", "NOOP")] += 1
    denom = max(1, total)
    return {
        "em": em / denom,
        "f1": f1 / denom,
        "accuracy": acc / denom,
        "avg_llm_calls": sum(e.get("usage", {}).get("llm_calls", 0) for e in episodes) / denom,
        "avg_retriever_calls": sum(e.get("usage", {}).get("retriever_calls", 0) for e in episodes) / denom,
        "avg_answer_rounds": sum(len(e.get("answer_stage", {}).get("rounds", [])) for e in episodes) / denom,
        "stop_reason_counts": dict(stop_counts),
        "memory_ops_counts": dict(op_counts),
        "dtcls_lesson_count": sum(len(e.get("dtcls", {}).get("lessons", [])) for e in episodes),
        "crds_drift_count": sum(1 for e in episodes if (e.get("crds", {}).get("drift_score") or 0) >= 0.25),
    }


def select_items(items, args):
    if args.run_mode == "single":
        return [(args.sample_id, items[args.sample_id])]
    indices = list(range(args.start_idx, args.end_idx if args.end_idx is not None else len(items)))
    if args.random_sample:
        random.seed(args.seed)
        random.shuffle(indices)
    if args.sample_num is not None:
        indices = indices[: args.sample_num]
    return [(idx, items[idx]) for idx in indices]


def main():
    dotenv.load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    cfg = load_config(args.config)
    dataset_name = canonical_name(args.dataset_name or "question")
    items = []
    if args.run_mode != "question":
        dataset_name, folder = resolve_dataset_folder(args.data_dir, args.dataset_name)
        split_path = folder / f"{args.split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        items = load_jsonl(split_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_mode": args.run_mode,
                    "dataset_name": dataset_name,
                    "items": len(items),
                    "model": args.openai_model or os.getenv("OPENAI_MODEL") or cfg.get("openai", {}).get("model"),
                    "memory_enabled": not args.no_memory,
                    "dtcls_enabled": not args.no_dtcls,
                    "crds_enabled": not args.no_crds,
                },
                indent=2,
            )
        )
        return
    output_dir = build_output_dir(args, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = make_generator(args, cfg)
    memory = None if args.no_memory else MemoryBank(args.memory_path)
    retriever = MetadataRetriever(items, topk=3)
    memory_cfg = cfg.get("memory", {})
    memory_cfg["enabled"] = not args.no_memory
    signals_cfg = cfg.get("signals", {})
    signals_cfg["dtcls_enabled"] = not args.no_dtcls
    signals_cfg["crds_enabled"] = not args.no_crds
    pipeline = DRAGSEAPipeline(
        cfg,
        generator=generator,
        retriever=retriever,
        memory_bank=memory,
        max_query_debate_rounds=args.max_query_debate_rounds,
        max_answer_debate_rounds=args.max_answer_debate_rounds,
        adaptive_config=cfg.get("adaptive_stopping", {}),
        memory_config=memory_cfg,
        signals_config=signals_cfg,
        save_prompts=args.save_prompts or cfg.get("run", {}).get("save_prompts", False),
        dataset_name=dataset_name,
    )
    episodes = []
    errors = []
    if args.run_mode == "question":
        if not args.question:
            raise ValueError("--question is required for question mode.")
        episode = pipeline.run_question(args.question)
        episodes.append(episode)
    else:
        for sample_id, item in select_items(items, args):
            try:
                episodes.append(pipeline.run_one(item, sample_id=sample_id))
            except Exception as exc:
                error = {
                    "sample_id": sample_id,
                    "question": item.get("question"),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "created_at": datetime.now().isoformat(),
                }
                errors.append(error)
                if not args.continue_on_error:
                    write_jsonl(output_dir / "errors.jsonl", [error])
                    raise
    write_jsonl(output_dir / "episodes.jsonl", episodes)
    write_jsonl(output_dir / "predictions.jsonl", [prediction_from_episode(ep) for ep in episodes])
    memory_ops = []
    for episode in episodes:
        memory_ops.extend(episode.get("memory_ops", []))
    write_jsonl(output_dir / "memory_ops.jsonl", memory_ops)
    if errors:
        write_jsonl(output_dir / "errors.jsonl", errors)
    metrics = compute_metrics(episodes)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    if episodes:
        first = episodes[0]
        print(f"Question: {first.get('question')}")
        print(f"Gold answers: {first.get('golden_answers')}")
        print(f"Final answer: {first.get('answer_stage', {}).get('final_answer')}")
        print(f"Stop reason: {first.get('answer_stage', {}).get('stop_reason')}")
        print(f"Query pool: {list(first.get('query_stage', {}).get('final_query_pool', {}).keys())}")
        print(f"Memory ops count: {len(first.get('memory_ops', []))}")
    print(f"Output path: {output_dir}")


if __name__ == "__main__":
    main()
