# COMMANDS

## 1) Setup (PowerShell)

```powershell
cd "d:\Learning\Research P-L\AAMAS\Self-Evolving_RAG"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r "..\Debate-Augmented-RAG\requirements.txt"
```

## 2) Build Tiny Local Dataset (Optional)

```powershell
cd "d:\Learning\Research P-L\AAMAS\Self-Evolving_RAG"
python .\sample_dataset.py
```

## 3) Prepare Retrieval Files (Required for RAG methods)

Put these files under `.\wiki_corpus\`:

- `wiki18_100w.jsonl`
- `e5_flat_inner.index`

Then run:

```powershell
Get-ChildItem .\wiki_corpus\
```

## 4) Quick Run (Default FlashRAG Config/Framework)

```powershell
cd "d:\Learning\Research P-L\AAMAS\Self-Evolving_RAG"
python .\main.py --method_name naive_gen --dataset_name strategyqa --split dev
```

## 5) Run with Ollama (Local LLM)

```powershell
cd "d:\Learning\Research P-L\AAMAS\Self-Evolving_RAG"
python .\main.py `
  --method_name drag `
  --dataset_name 2wikimultihopqa `
  --split dev `
  --llm_provider ollama `
  --generator_model llama3.1:8b `
  --ollama_base_url http://localhost:11434/v1
```

## 6) Run with OpenAI API

Option A: pass key directly

```powershell
cd "d:\Learning\Research P-L\AAMAS\Self-Evolving_RAG"
python .\main.py `
  --method_name naive_rag `
  --dataset_name nq `
  --split dev `
  --llm_provider openai `
  --generator_model gpt-4o-mini `
  --openai_api_key "<YOUR_OPENAI_API_KEY>"
```

Option B: use environment variable

```powershell
$env:OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY>"
python .\main.py --method_name naive_rag --dataset_name nq --split dev --llm_provider openai --generator_model gpt-4o-mini
```

## 7) Run Each Baseline Method

```powershell
python .\main.py --method_name naive_gen   --dataset_name strategyqa --split dev
python .\main.py --method_name naive_rag   --dataset_name strategyqa --split dev
python .\main.py --method_name flare       --dataset_name strategyqa --split dev
python .\main.py --method_name iterretgen  --dataset_name strategyqa --split dev
python .\main.py --method_name ircot       --dataset_name strategyqa --split dev
python .\main.py --method_name self_ask    --dataset_name strategyqa --split dev
python .\main.py --method_name sure        --dataset_name strategyqa --split dev
python .\main.py --method_name selfrag     --dataset_name strategyqa --split dev
python .\main.py --method_name retrobust   --dataset_name strategyqa --split dev
python .\main.py --method_name mad         --dataset_name strategyqa --split dev
python .\main.py --method_name drag        --dataset_name strategyqa --split dev
```

## 8) Debate-Specific Controls

```powershell
python .\main.py `
  --method_name drag `
  --dataset_name strategyqa `
  --split dev `
  --max_query_debate_rounds 3 `
  --max_answer_debate_rounds 3 `
  --query_proponent_agent 1 `
  --query_opponent_agent 1 `
  --answer_proponent_agent 1 `
  --answer_opponent_agent 1
```

MAD-specific:

```powershell
python .\main.py --method_name mad --dataset_name strategyqa --split dev --agents 2 --rag_agents 0
```

## 9) Useful Overrides

```powershell
python .\main.py `
  --method_name drag `
  --dataset_name nq `
  --split test `
  --test_sample_num 10 `
  --save_dir .\output
```

## 10) Outputs

Run outputs are saved under:

```text
Self-Evolving_RAG/output/<DatasetName>/<method_name>/
```
