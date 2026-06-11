#!/usr/bin/env bash
set -euo pipefail

# End-to-end setup for FlashRAG/DRAG retrieval baselines.
#
# Usage:
#   bash scripts/setup_drag_single.sh
#
# Optional environment variables:
#   PYTHON_BIN=python3
#   VENV_DIR=.venv
#   SKIP_VENV=1
#   SKIP_REQUIREMENTS=1
#   SKIP_TINY_DATA=1
#   SKIP_E5=1
#   SKIP_CORPUS=1
#   E5_MODEL_ID=intfloat/e5-base-v2
#   MODELSCOPE_DATASET_ID=hhjinjiajie/FlashRAG_Dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
E5_MODEL_ID="${E5_MODEL_ID:-intfloat/e5-base-v2}"
E5_DIR="${E5_DIR:-${PROJECT_ROOT}/models/e5-base-v2}"
MODELSCOPE_DATASET_ID="${MODELSCOPE_DATASET_ID:-hhjinjiajie/FlashRAG_Dataset}"
WIKI_DIR="${WIKI_DIR:-${PROJECT_ROOT}/wiki_corpus}"
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-${PROJECT_ROOT}/config/local_setup_overrides.json}"

log() {
  printf '\n[setup] %s\n' "$*"
}

warn() {
  printf '\n[setup][warn] %s\n' "$*" >&2
}

die() {
  printf '\n[setup][error] %s\n' "$*" >&2
  exit 1
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    die "Python was not found. Set PYTHON_BIN=/path/to/python and rerun."
  fi
fi

cd "${PROJECT_ROOT}"
mkdir -p "${WIKI_DIR}" "$(dirname "${CONFIG_OVERRIDE}")" "${PROJECT_ROOT}/models"

if [ "${SKIP_VENV:-0}" != "1" ]; then
  if [ ! -d "${VENV_DIR}" ]; then
    log "Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  if [ -f "${VENV_DIR}/bin/activate" ]; then
    # Linux/macOS/WSL
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
    # Git Bash on Windows
    # shellcheck disable=SC1091
    source "${VENV_DIR}/Scripts/activate"
  else
    die "Could not find an activation script inside ${VENV_DIR}."
  fi
fi

log "Using Python: $(command -v python)"
python -m pip install -U pip setuptools wheel

if [ "${SKIP_REQUIREMENTS:-0}" != "1" ]; then
  log "Installing project requirements"
  python -m pip install -r requirements.txt
fi

log "Installing download helpers"
python -m pip install -U huggingface_hub modelscope datasets

if [ "${SKIP_TINY_DATA:-0}" != "1" ]; then
  log "Downloading tiny FlashRAG datasets into data/flashrag_tiny"
  python sample_dataset.py
fi

if [ "${SKIP_E5:-0}" != "1" ]; then
  if [ -f "${E5_DIR}/config.json" ]; then
    log "E5 model already exists at ${E5_DIR}"
  else
    log "Downloading E5 embedding model ${E5_MODEL_ID}"
    python - "${E5_MODEL_ID}" "${E5_DIR}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=model_id,
    local_dir=str(target),
    local_dir_use_symlinks=False,
)
print(f"Downloaded {model_id} to {target}")
PY
  fi
fi

find_downloaded_file() {
  local search_dir="$1"
  local basename="$2"
  find "${search_dir}" -type f -name "${basename}" -print -quit 2>/dev/null || true
}

download_modelscope_file() {
  local remote_path="$1"
  local dest_path="$2"
  local basename
  local tmp_dir
  basename="$(basename "${remote_path}")"
  tmp_dir="${WIKI_DIR}/.modelscope_download"

  if [ -f "${dest_path}" ]; then
    log "${basename} already exists at ${dest_path}"
    return 0
  fi

  mkdir -p "${tmp_dir}"
  log "Downloading ${remote_path} from ModelScope dataset ${MODELSCOPE_DATASET_ID}"

  if command -v modelscope >/dev/null 2>&1; then
    if modelscope download --dataset "${MODELSCOPE_DATASET_ID}" "${remote_path}" --local_dir "${tmp_dir}"; then
      :
    else
      warn "ModelScope CLI download failed for ${remote_path}; trying Python SDK fallback."
    fi
  fi

  local found
  found="$(find_downloaded_file "${tmp_dir}" "${basename}")"

  if [ -z "${found}" ]; then
    python - "${MODELSCOPE_DATASET_ID}" "${remote_path}" "${tmp_dir}" <<'PY' || true
import sys
from pathlib import Path

dataset_id, remote_path, tmp_dir = sys.argv[1:4]
Path(tmp_dir).mkdir(parents=True, exist_ok=True)

try:
    from modelscope import snapshot_download
except Exception:
    from modelscope.hub.snapshot_download import snapshot_download

attempts = [
    {
        "model_id": dataset_id,
        "repo_type": "dataset",
        "allow_file_pattern": remote_path,
        "local_dir": tmp_dir,
    },
    {
        "model_id": dataset_id,
        "repo_type": "dataset",
        "allow_patterns": remote_path,
        "local_dir": tmp_dir,
    },
    {
        "model_id": dataset_id,
        "allow_file_pattern": remote_path,
        "local_dir": tmp_dir,
    },
]

last_error = None
for kwargs in attempts:
    try:
        snapshot_download(**kwargs)
        print(f"Downloaded {remote_path}")
        break
    except TypeError as exc:
        last_error = exc
    except Exception as exc:
        last_error = exc
else:
    raise RuntimeError(f"Could not download {remote_path}: {last_error}")
PY
    found="$(find_downloaded_file "${tmp_dir}" "${basename}")"
  fi

  if [ -z "${found}" ]; then
    return 1
  fi

  cp "${found}" "${dest_path}"
  log "Placed ${basename} at ${dest_path}"
}

if [ "${SKIP_CORPUS:-0}" != "1" ]; then
  download_modelscope_file "retrieval_corpus/wiki18_100w.jsonl" "${WIKI_DIR}/wiki18_100w.jsonl" || {
    warn "Could not automatically download wiki18_100w.jsonl."
    warn "Manual source: https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files"
  }

  download_modelscope_file "retrieval_corpus/e5_flat_inner.index" "${WIKI_DIR}/e5_flat_inner.index" || {
    warn "Could not automatically download e5_flat_inner.index."
    warn "Manual source: https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files"
  }
fi

log "Writing local config override to ${CONFIG_OVERRIDE}"
python - "${PROJECT_ROOT}" "${E5_DIR}" "${WIKI_DIR}" "${CONFIG_OVERRIDE}" <<'PY'
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
e5_dir = Path(sys.argv[2]).resolve()
wiki_dir = Path(sys.argv[3]).resolve()
config_override = Path(sys.argv[4]).resolve()

payload = {
    "model2path": {
        "e5": str(e5_dir),
        "llama3-8B-instruct": None,
        "llama3.2-3B-instruct": None,
        "llama2-7B": None,
        "qwen2.5-14B": None,
    },
    "retrieval_method": "e5",
    "index_path": str(wiki_dir / "e5_flat_inner.index"),
    "corpus_path": str(wiki_dir / "wiki18_100w.jsonl"),
}

config_override.parent.mkdir(parents=True, exist_ok=True)
config_override.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(config_override)
PY

missing=0
[ -f "${E5_DIR}/config.json" ] || { warn "Missing E5 model at ${E5_DIR}"; missing=1; }
[ -f "${WIKI_DIR}/wiki18_100w.jsonl" ] || { warn "Missing ${WIKI_DIR}/wiki18_100w.jsonl"; missing=1; }
[ -f "${WIKI_DIR}/e5_flat_inner.index" ] || { warn "Missing ${WIKI_DIR}/e5_flat_inner.index"; missing=1; }

if [ "${missing}" -ne 0 ]; then
  die "Setup finished with missing retrieval assets. Download the missing files manually or rerun with network access."
fi

log "Setup complete."
cat <<EOF

Try DRAG single with OpenAI:
  OPENAI_API_KEY=... python main.py \\
    --method_name drag_single \\
    --dataset_name strategyqa \\
    --split dev \\
    --test_sample_num 10 \\
    --llm_provider openai \\
    --generator_model gpt-4o-mini \\
    --config_json "${CONFIG_OVERRIDE}"

Try DRAG single with Ollama:
  python main.py \\
    --method_name drag_single \\
    --dataset_name strategyqa \\
    --split dev \\
    --test_sample_num 10 \\
    --llm_provider ollama \\
    --generator_model llama3.1:8b \\
    --config_json "${CONFIG_OVERRIDE}"
EOF
