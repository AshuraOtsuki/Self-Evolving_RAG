import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (  # noqa: E402
    build_config,
    ensure_dataset_alias_folder,
    normalize_dataset_name,
    normalize_method_name,
    patch_flashrag_cpu_retriever,
)
from model.baselines.DRAG_single import QueryDebateSingleAnswerRAG  # noqa: E402
from model.baselines.drag_modules import generate_single, render_messages  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug DRAG_SINGLE one module at a time: dataset, retrieve, prompt, generate."
    )
    parser.add_argument(
        "--step",
        choices=["all", "config", "dataset", "retrieve", "prompt", "generate", "query_round", "answer"],
        default="all",
        help="Last step to execute. Earlier prerequisite steps are also run.",
    )
    parser.add_argument("--base_config", default=str(REPO_ROOT / "config" / "ollama_base_config.yaml"))
    parser.add_argument("--dataset_name", default="strategyqa")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "flashrag_tiny"))
    parser.add_argument("--test_sample_num", type=int, default=1)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--question", default=None)
    parser.add_argument("--save_dir", default=str(REPO_ROOT / "output" / "debug_modules"))
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--generator_model", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--framework", default=None)
    parser.add_argument("--retrieval_method", default=None)
    parser.add_argument("--retrieval_model_path", default=None)
    parser.add_argument("--index_path", default=None)
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--generator_batch_size", type=int, default=None)
    parser.add_argument("--max_query_debate_rounds", type=int, default=1)
    parser.add_argument("--max_answer_debate_rounds", type=int, default=0)
    parser.add_argument("--query_proponent_agent", type=int, default=1)
    parser.add_argument("--query_opponent_agent", type=int, default=1)
    parser.add_argument("--answer_proponent_agent", type=int, default=1)
    parser.add_argument("--answer_opponent_agent", type=int, default=1)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--rag_agents", type=int, default=0)
    parser.add_argument("--config_json", default=None)
    parser.add_argument("--llm_provider", choices=["default", "ollama", "openai"], default="default")
    parser.add_argument("--openai_base_url", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--ollama_base_url", default="http://localhost:11434/v1")
    parser.add_argument("--ollama_api_key", default="ollama")
    parser.add_argument("--debug_preview_chars", type=int, default=600)
    parser.add_argument("--show_prompt", action="store_true")
    return parser.parse_args()


def should_stop(args, step):
    order = ["config", "dataset", "retrieve", "prompt", "generate", "query_round", "answer"]
    return args.step != "all" and order.index(step) >= order.index(args.step)


def preview(value, limit=600):
    text = "" if value is None else str(value)
    compact = " ".join(text.split())
    if not compact:
        return "<empty>" if text == "" else f"<whitespace chars={len(text)}>"
    return compact[:limit] + ("..." if len(compact) > limit else "")


def cfg_get(cfg, key, default=None):
    try:
        return cfg[key]
    except Exception:
        return default


def print_json(title, data):
    print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def summarize_docs(docs):
    rows = []
    for idx, doc in enumerate(docs or [], start=1):
        contents = doc.get("contents", "")
        title = contents.split("\n", 1)[0] if contents else doc.get("title", "")
        rows.append(
            {
                "rank": idx,
                "id": doc.get("id"),
                "score": doc.get("score"),
                "title": title,
                "preview": preview(contents, 300),
            }
        )
    return rows


def make_main_args(args, method_name, dataset_name):
    values = vars(args).copy()
    values["method_name"] = method_name
    values["dataset_name"] = dataset_name
    values["debug_steps"] = True
    values["do_eval"] = False
    return SimpleNamespace(**values)


def select_item(args, dataset):
    if args.question:
        item = dataset[args.sample_index]
        item.question = args.question
        return item
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"--sample_index must be between 0 and {len(dataset) - 1}")
    return dataset[args.sample_index]


def main():
    args = parse_args()
    method_name = normalize_method_name("drag_single")
    dataset_name = normalize_dataset_name(args.dataset_name)
    main_args = make_main_args(args, method_name, dataset_name)

    ensure_dataset_alias_folder(args.data_dir, dataset_name)
    cfg = build_config(main_args, method_name, dataset_name)
    patch_flashrag_cpu_retriever()

    print_json(
        "CONFIG",
        {
            "method_name": cfg_get(cfg, "method_name"),
            "dataset_name": cfg_get(cfg, "dataset_name"),
            "framework": cfg_get(cfg, "framework"),
            "generator_model": cfg_get(cfg, "generator_model"),
            "generator_model_path": cfg_get(cfg, "generator_model_path"),
            "retrieval_method": cfg_get(cfg, "retrieval_method"),
            "retrieval_model_path": cfg_get(cfg, "retrieval_model_path"),
            "index_path": cfg_get(cfg, "index_path"),
            "corpus_path": cfg_get(cfg, "corpus_path"),
        },
    )
    if should_stop(args, "config"):
        return

    from flashrag.utils import get_dataset

    all_splits = get_dataset(cfg)
    dataset = all_splits[args.split]
    item = select_item(args, dataset)
    question = item.question
    print_json(
        "DATASET ITEM",
        {
            "sample_index": args.sample_index,
            "id": getattr(item, "id", None),
            "question": question,
            "golden_answers": getattr(item, "golden_answers", None),
        },
    )
    if should_stop(args, "dataset"):
        return

    pipeline = QueryDebateSingleAnswerRAG(
        cfg,
        max_query_debate_rounds=args.max_query_debate_rounds,
        query_proponent_agent=args.query_proponent_agent,
        query_opponent_agent=args.query_opponent_agent,
    )

    docs = pipeline.retriever.search(question)
    query_pool = {question.strip(): docs}
    print_json("RETRIEVE", {"query": question, "doc_count": len(docs), "docs": summarize_docs(docs)})
    if should_stop(args, "retrieve"):
        return

    proponent_message = [
        pipeline.prompt_builder_module.query_stage_system_message("Proponent Agent 0"),
        {
            "role": "user",
            "content": f"Question: {question}\n{pipeline.query_pool_module.format_query_pool(query_pool)}",
        },
    ]
    proponent_prompt = render_messages(proponent_message, cfg)
    print_json(
        "PROPONENT PROMPT",
        {
            "type": type(proponent_prompt).__name__,
            "message_count": len(proponent_prompt) if isinstance(proponent_prompt, list) else None,
            "preview": preview(proponent_prompt, args.debug_preview_chars),
        },
    )
    if args.show_prompt:
        print_json("PROPONENT PROMPT FULL", proponent_prompt)
    if should_stop(args, "prompt"):
        return

    proponent_output = generate_single(pipeline.generator, proponent_prompt, cfg)
    print_json(
        "PROPONENT GENERATE",
        {
            "chars": len(str(proponent_output or "")),
            "repr": repr(proponent_output),
            "preview": preview(proponent_output, args.debug_preview_chars),
        },
    )
    if should_stop(args, "generate"):
        return

    opponent_message = [
        pipeline.prompt_builder_module.query_stage_system_message("Opponent Agent 0"),
        {
            "role": "user",
            "content": f"Question: {question}\n{pipeline.query_pool_module.format_query_pool(query_pool)}",
        },
    ]
    opponent_output = generate_single(pipeline.generator, render_messages(opponent_message, cfg), cfg)
    moderator_message = [
        pipeline.prompt_builder_module.query_stage_moderator_message(
            {
                "Proponent Agent 0": [proponent_prompt, proponent_output],
                "Opponent Agent 0": [render_messages(opponent_message, cfg), opponent_output],
            },
            question,
            query_pool,
        )
    ]
    moderator_output = generate_single(pipeline.generator, render_messages(moderator_message, cfg), cfg)
    print_json(
        "QUERY ROUND",
        {
            "opponent_repr": repr(opponent_output),
            "opponent_preview": preview(opponent_output, args.debug_preview_chars),
            "moderator_repr": repr(moderator_output),
            "moderator_preview": preview(moderator_output, args.debug_preview_chars),
        },
    )
    if should_stop(args, "query_round"):
        return

    answer_message = [
        pipeline.prompt_builder_module.answer_only_message(query_pool),
        {"role": "user", "content": f"Question: {question}\n"},
    ]
    answer_prompt = render_messages(answer_message, cfg)
    if args.show_prompt:
        print_json("ANSWER PROMPT FULL", answer_prompt)
    answer_output = generate_single(pipeline.generator, answer_prompt, cfg)
    print_json(
        "ANSWER",
        {
            "raw_repr": repr(answer_output),
            "raw_preview": preview(answer_output, args.debug_preview_chars),
            "parsed": pipeline._parse_answer(answer_output),
        },
    )


if __name__ == "__main__":
    main()
