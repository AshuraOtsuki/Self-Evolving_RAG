# 00 — Context and Target

## 1. Background

DRAG gốc là một framework RAG kết hợp Multi-Agent Debate vào 2 stage:

1. **Retrieval Debate**: Proponent bảo vệ query hiện tại và retrieved docs; Challenger/Opponent phê phán query, đề xuất query optimization hoặc query expansion; Judge/Moderator quyết định giữ query hay update query pool.
2. **Response Debate**: Proponent trả lời dựa trên retrieved docs; Challenger/Opponent trả lời dựa trên own knowledge; Judge chọn final answer.

Codebase gốc có các file quan trọng:

```text
main.py
config/base_config.yaml
config/init_config.py
model/drag.py
model/prompt.py
model/baseline.py
model/mad.py
```

Trong `main.py`, method `DRAG` khởi tạo `DebateAugmentedRAG` với các tham số:

```python
max_query_debate_rounds
max_answer_debate_rounds
query_opponent_agent
query_proponent_agent
answer_opponent_agent
answer_proponent_agent
```

Trong `model/drag.py`, class `DebateAugmentedRAG` hiện dùng:

```python
self.generator = get_generator(config)
self.retriever = get_retriever(config)
```

và gọi:

```python
self.generator.generate(input_prompt)[0]
self.retriever.search(query)
```

Điểm tích hợp dễ nhất: tạo một generator adapter có method `.generate(prompts)` tương thích FlashRAG generator interface, hoặc inject `generator=OpenAIGenerator(...)` vào pipeline mới.

## 2. Target Framework

Tên framework: **DRAG+SEA** hoặc code name `DRAG_SEA`.

SEA = Self-Evolving Architecture.

DRAG+SEA mở rộng DRAG bằng 3 phần:

### 2.1 Memory-Guided Debate

Trước mỗi episode, retrieve top-k relevant lessons từ Memory Bank. Lessons được inject vào prompt của:

- Query Proponent
- Query Challenger/Opponent
- Query Judge
- Answer Proponent
- Answer Challenger/Opponent
- Answer Judge

### 2.2 DTCLS

Sau mỗi episode, extract lessons từ debate transcript. Lesson tập trung vào tactic nào giúp retrieval/generation thành công hoặc thất bại.

### 2.3 CRDS

Sau mỗi episode, lưu retrieval snapshot. Khi query mới liên quan đến entity/topic cũ, so sánh retrieved docs mới với old snapshots để detect drift. Nếu drift cao, trigger UPDATE/DELETE/DOWNWEIGHT memory.

### 2.4 Adaptive Response Debate

Thay vì chạy cố định `max_answer_debate_rounds`, sau mỗi round tính:

- agreement score
- evidence support score
- answer stability score
- drift risk score

Nếu đủ điều kiện thì dừng sớm.

## 3. Non-goals for v1

Không cần:

- fine-tuning model
- RL
- vector database production-grade
- UI
- async distributed execution
- reranker mới
- custom retrieval corpus mới

## 4. Expected CLI examples

Chạy 1 sample theo index:

```bash
python scripts/run_drag_sea.py \
  --dataset_name 2wiki \
  --data_dir ./data/flashrag_data \
  --run_mode single \
  --sample_id 0 \
  --openai_model gpt-4.1-mini \
  --memory_path ./output/drag_sea/memory.sqlite
```

Chạy 1 câu hỏi custom:

```bash
python scripts/run_drag_sea.py \
  --run_mode question \
  --question "Who does Fez marry in That '70s Show?" \
  --openai_model gpt-4.1-mini
```

Chạy batch:

```bash
python scripts/run_drag_sea.py \
  --dataset_name strategyqa \
  --data_dir ./data/flashrag_data \
  --run_mode batch \
  --sample_num 100 \
  --start_idx 0 \
  --openai_model gpt-4.1-mini
```
