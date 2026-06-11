# DRAG+SEA Implementation Specs

Bộ specs này dùng để giao cho coding agents implement **DRAG+SEA: Self-Evolving Debate-Augmented RAG** trên codebase gốc `Huenao/Debate-Augmented-RAG`.

Mục tiêu: giữ backbone DRAG gốc, thay local HF/vLLM generator bằng OpenAI API adapter, thêm self-evolving memory qua 2 signals:

- **DTCLS**: Debate-Trajectory Contrastive Lesson Signal
- **CRDS**: Cross-Session Retrieval Drift Signal
- **Adaptive Response Debate**: dừng sớm response debate khi đủ evidence / agreement hoặc phát hiện drift risk.

## Thứ tự đọc spec

1. `00_CONTEXT_AND_TARGET.md`
2. `01_TARGET_ARCHITECTURE.md`
3. `02_REPO_INTEGRATION_PLAN.md`
4. `03_OPENAI_API_ADAPTER.md`
5. `04_DATA_AND_RUN_MODES.md`
6. `05_MEMORY_BANK.md`
7. `06_DTCLS_SIGNAL.md`
8. `07_CRDS_SIGNAL.md`
9. `08_ADAPTIVE_RESPONSE_DEBATE.md`
10. `09_LOGGING_OUTPUTS_EVAL.md`
11. `10_AGENT_TASK_BREAKDOWN.md`

## Deliverable coding cuối cùng

Coding agent cần tạo được:

```text
Debate-Augmented-RAG/
├── model/
│   ├── drag_sea.py
│   ├── openai_generator.py
│   ├── memory_bank.py
│   ├── lesson_retriever.py
│   ├── dtcls.py
│   ├── crds.py
│   ├── adaptive_stopping.py
│   └── schemas.py
├── config/
│   └── drag_sea_config.yaml
├── prompts/
│   └── drag_sea_prompts.py
├── scripts/
│   ├── run_drag_sea.py
│   ├── inspect_memory.py
│   └── export_episode_logs.py
├── tests/
│   ├── test_openai_generator.py
│   ├── test_memory_bank.py
│   ├── test_dtcls.py
│   ├── test_crds.py
│   └── test_adaptive_stopping.py
└── output/
```

## Nguyên tắc implement

- Không phá DRAG baseline. Thêm method mới `DRAG_SEA` hoặc script riêng `scripts/run_drag_sea.py`.
- Giữ retriever của FlashRAG nếu repo đã chạy được retrieval/index.
- OpenAI API chỉ thay phần generator/debate/lesson extraction.
- Mọi output quan trọng phải log JSONL để debug.
- Memory Bank ban đầu dùng SQLite + JSON fields. Không cần vector DB phức tạp trong version đầu.
- Có mode chạy 1 sample để debug prompt/cost và mode batch nhiều sample để eval.
