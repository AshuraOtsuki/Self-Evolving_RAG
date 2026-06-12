import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file if present

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHOD_NAME_MAP = {
    "drag": "DRAG",
    "drag_single": "DRAG_SINGLE",
    "drag_single_answer": "DRAG_SINGLE",
    "drag_query_single": "DRAG_SINGLE",
    "naive_gen": "naive_gen",
    "naive_rag": "naive_rag",
    "flare": "flare",
    "iterretgen": "iterretgen",
    "iter_retgen": "iterretgen",
    "ircot": "ircot",
    "self_ask": "self_ask",
    "sure": "sure",
    "selfrag": "selfrag",
    "self_rag": "selfrag",
    "retrobust": "retrobust",
    "ret_robust": "retrobust",
    "mad": "mad",
}

DATASET_NAME_MAP = {
    "nq": "NQ",
    "2wiki": "2wiki",
    "2wikimultihopqa": "2wiki",
    "strategyqa": "StrategyQA",
}

DATASET_FOLDER_CANDIDATES = {
    "NQ": ["NQ", "nq"],
    "2wiki": ["2wiki", "2wikimultihopqa", "2WikiMultiHopQA"],
    "StrategyQA": ["StrategyQA", "strategyqa"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DRAG/MAD/baseline methods in Self-Evolving_RAG with local FlashRAG-format data."
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="naive_gen",
        help=(
            "Method to run. Supported: drag, drag_single, naive_gen, naive_rag, flare, iterretgen, "
            "ircot, self_ask, sure, selfrag, retrobust, mad."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="strategyqa",
        help="Dataset name or alias. Supported aliases: nq, 2wiki, 2wikimultihopqa, strategyqa.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "flashrag_tiny"),
        help="Directory containing FlashRAG dataset folders.",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=str((REPO_ROOT / "config" / "base_config.yaml").resolve()),
        help="Path to base FlashRAG yaml config.",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Optional JSON file with extra config overrides.",
    )
    parser.add_argument("--test_sample_num", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default=str(REPO_ROOT / "output"))
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--generator_model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--framework", type=str, default=None)
    parser.add_argument("--retrieval_method", type=str, default=None)
    parser.add_argument("--retrieval_model_path", type=str, default=None)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str, default=None)
    parser.add_argument("--generator_batch_size", type=int, default=None)
    parser.add_argument("--max_query_debate_rounds", type=int, default=3)
    parser.add_argument("--max_answer_debate_rounds", type=int, default=3)
    parser.add_argument("--query_proponent_agent", type=int, default=1)
    parser.add_argument("--query_opponent_agent", type=int, default=1)
    parser.add_argument("--answer_proponent_agent", type=int, default=1)
    parser.add_argument("--answer_opponent_agent", type=int, default=1)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--rag_agents", type=int, default=0)
    parser.add_argument("--do_eval", action="store_true")

    # LLM provider selection (default keeps the existing FlashRAG config behavior).
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="default",
        choices=["default", "ollama", "openai"],
        help=(
            "LLM backend mode. "
            "'default' uses your existing FlashRAG config/framework. "
            "'ollama' and 'openai' both use FlashRAG openai framework with different base_url/api_key."
        ),
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=None,
        help="Optional custom OpenAI-compatible base URL for API provider.",
    )
    parser.add_argument(
        "--ollama_base_url",
        type=str,
        default="http://localhost:11434/v1",
        help="Ollama OpenAI-compatible endpoint base URL.",
    )
    parser.add_argument(
        "--ollama_api_key",
        type=str,
        default="ollama",
        help="API key sent to Ollama-compatible endpoint (typically any non-empty string).",
    )
    return parser.parse_args()


def normalize_method_name(raw_name):
    key = raw_name.strip().lower().replace('"', "").replace("'", "")
    key = key.replace(" ", "_").replace("-", "_")
    if key not in METHOD_NAME_MAP:
        raise ValueError(f"Unsupported method_name: {raw_name}")
    return METHOD_NAME_MAP[key]


def normalize_dataset_name(raw_name):
    key = raw_name.strip().lower().replace('"', "").replace("'", "")
    if key not in DATASET_NAME_MAP:
        raise ValueError(f"Unsupported dataset_name: {raw_name}")
    return DATASET_NAME_MAP[key]


def ensure_dataset_alias_folder(data_dir, canonical_dataset_name):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = data_dir / canonical_dataset_name
    if canonical_dir.exists():
        return

    src_dir = None
    for candidate in DATASET_FOLDER_CANDIDATES[canonical_dataset_name]:
        candidate_dir = data_dir / candidate
        if candidate_dir.exists():
            src_dir = candidate_dir
            break

    if src_dir is None:
        candidates = ", ".join(DATASET_FOLDER_CANDIDATES[canonical_dataset_name])
        raise FileNotFoundError(
            f"Dataset folder for '{canonical_dataset_name}' not found under '{data_dir}'. "
            f"Tried: {candidates}"
        )

    canonical_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        src_file = src_dir / f"{split}.jsonl"
        if src_file.exists():
            shutil.copy2(src_file, canonical_dir / src_file.name)


def load_extra_config(config_json_path):
    if config_json_path is None:
        return {}
    with open(config_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_flashrag_basic_config(base_config_path):
    """
    Some flashrag pip builds miss `flashrag/config/basic_config.yaml`.
    The Config class hard-requires it, so create it from our base config if absent.
    """
    try:
        import flashrag.config as flashrag_config_pkg
    except Exception:
        return

    if not hasattr(flashrag_config_pkg, "__file__"):
        return

    pkg_dir = Path(flashrag_config_pkg.__file__).resolve().parent
    basic_config_path = pkg_dir / "basic_config.yaml"
    if basic_config_path.exists():
        return

    shutil.copy2(base_config_path, basic_config_path)
    print(f"[INFO] Created missing flashrag basic config: {basic_config_path}")


def patch_flashrag_cpu_retriever():
    """
    Older FlashRAG builds unconditionally call model.cuda() for dense retrievers.
    Patch that loader when the current PyTorch build cannot execute CUDA kernels.
    """
    try:
        import torch
    except Exception:
        return

    cuda_error = None
    if torch.cuda.is_available():
        try:
            probe = torch.ones(1, device="cuda")
            _ = (probe + 1).item()
            return
        except Exception as exc:
            cuda_error = exc

    else:
        cuda_error = "CUDA is not available to PyTorch"

    if cuda_error is None:
        return

    try:
        from transformers import AutoModel, AutoTokenizer
        import flashrag.retriever.utils as retriever_utils
    except Exception as exc:
        print(f"[WARN] Could not patch FlashRAG CPU retriever loader: {exc}")
        return

    def load_model_cpu(model_path, use_fp16=False):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        model.to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        return model, tokenizer

    retriever_utils.load_model = load_model_cpu
    try:
        import flashrag.retriever.encoder as retriever_encoder

        retriever_encoder.load_model = load_model_cpu
    except Exception:
        pass

    print(f"[INFO] Patched FlashRAG dense retriever loader for CPU fallback: {cuda_error}")


def apply_llm_provider_config(args, overrides):
    provider = args.llm_provider
    if provider == "default":
        return

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "llm_provider=openai requires OPENAI_API_KEY environment variable to be set."
            )
        overrides["framework"] = "openai"
        openai_setting = {"api_key": api_key, "base_url": args.openai_base_url}
        overrides["openai_setting"] = openai_setting
        return

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible API at /v1.
        overrides["framework"] = "openai"
        overrides["openai_setting"] = {
            "api_key": args.ollama_api_key,
            "base_url": args.ollama_base_url,
        }
        if not overrides.get("generator_model"):
            raise ValueError(
                "llm_provider=ollama requires --generator_model "
                "(example: --generator_model llama3.1:8b)."
            )
        return


def build_config(args, canonical_method_name, canonical_dataset_name):
    from flashrag.config import Config

    base_config_path = Path(args.base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(
            f"base_config not found at: {base_config_path}. "
            "Pass a valid --base_config path."
        )

    ensure_flashrag_basic_config(base_config_path)

    save_dir = Path(args.save_dir).resolve() / canonical_dataset_name / canonical_method_name
    save_dir.mkdir(parents=True, exist_ok=True)

    overrides = {
        "method_name": canonical_method_name,
        "dataset_name": canonical_dataset_name,
        "split": [args.split],
        "data_dir": str(Path(args.data_dir).resolve()),
        "test_sample_num": args.test_sample_num,
        "save_note": canonical_method_name,
        "save_dir": str(save_dir),
        "gpu_id": args.gpu_id,
        "max_query_debate_rounds": args.max_query_debate_rounds,
        "max_answer_debate_rounds": args.max_answer_debate_rounds,
        "query_proponent_agent": args.query_proponent_agent,
        "query_opponent_agent": args.query_opponent_agent,
        "answer_proponent_agent": args.answer_proponent_agent,
        "answer_opponent_agent": args.answer_opponent_agent,
        "agents": args.agents,
        "rag_agents": args.rag_agents,
    }

    optional_fields = {
        "generator_model": args.generator_model,
        "model_path": args.model_path,
        "framework": args.framework,
        "retrieval_method": args.retrieval_method,
        "index_path": args.index_path,
        "corpus_path": args.corpus_path,
        "generator_batch_size": args.generator_batch_size,
    }
    for key, value in optional_fields.items():
        if value is not None:
            overrides[key] = value

    overrides.update(load_extra_config(args.config_json))
    apply_llm_provider_config(args, overrides)

    try:
        return Config(str(base_config_path), overrides)
    except TypeError:
        return Config(config_file_path=str(base_config_path), config_dict=overrides)


def run_method(cfg, method_name, dataset, do_eval):
    if method_name == "DRAG":
        from model.baselines.DRAG import DebateAugmentedRAG

        pipeline = DebateAugmentedRAG(
            cfg,
            max_query_debate_rounds=cfg["max_query_debate_rounds"],
            max_answer_debate_rounds=cfg["max_answer_debate_rounds"],
            query_proponent_agent=cfg["query_proponent_agent"],
            query_opponent_agent=cfg["query_opponent_agent"],
            answer_proponent_agent=cfg["answer_proponent_agent"],
            answer_opponent_agent=cfg["answer_opponent_agent"],
        )
        return pipeline.run(dataset, do_eval=do_eval)

    if method_name == "DRAG_SINGLE":
        from model.baselines.DRAG_single import QueryDebateSingleAnswerRAG

        pipeline = QueryDebateSingleAnswerRAG(
            cfg,
            max_query_debate_rounds=cfg["max_query_debate_rounds"],
            query_proponent_agent=cfg["query_proponent_agent"],
            query_opponent_agent=cfg["query_opponent_agent"],
        )
        return pipeline.run(dataset, do_eval=do_eval)

    from model.baselines.baselines import Baselines

    runner = Baselines(cfg)
    return runner.run(method_name, dataset, do_eval=do_eval)


def main():
    args = parse_args()
    canonical_method_name = normalize_method_name(args.method_name)
    canonical_dataset_name = normalize_dataset_name(args.dataset_name)

    ensure_dataset_alias_folder(args.data_dir, canonical_dataset_name)
    cfg = build_config(args, canonical_method_name, canonical_dataset_name)
    patch_flashrag_cpu_retriever()

    from flashrag.utils import get_dataset

    all_splits = get_dataset(cfg)
    if args.split not in all_splits:
        available = ", ".join(all_splits.keys())
        raise KeyError(f"Split '{args.split}' is not available. Available splits: {available}")

    test_data = all_splits[args.split]
    run_method(cfg, canonical_method_name, test_data, do_eval=args.do_eval)

    print(
        f"Completed method={canonical_method_name}, dataset={canonical_dataset_name}, split={args.split}."
    )
    print(f"Outputs directory: {cfg['save_dir']}")


if __name__ == "__main__":
    main()
