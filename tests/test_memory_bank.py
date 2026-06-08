from model.memory_bank import MemoryBank


def test_add_retrieve_and_mark_outdated(tmp_path):
    memory = MemoryBank(str(tmp_path / "memory.sqlite"))
    memory_id = memory.add_memory(
        {
            "lesson_type": "entity_disambiguation",
            "target_role": "query_challenger",
            "trigger_condition": "ambiguous band name",
            "recommended_action": "add entity type to query",
            "confidence": 0.8,
        }
    )
    lessons = memory.retrieve_lessons("Is The Police a law enforcement group?", top_k=5, min_score=0.0)
    assert any(lesson["memory_id"] == memory_id for lesson in lessons)
    memory.mark_outdated(memory_id, "test")
    lessons = memory.retrieve_lessons("Is The Police a law enforcement group?", top_k=5, min_score=0.0)
    assert all(lesson["memory_id"] != memory_id for lesson in lessons)


def test_save_episode_and_snapshot(tmp_path):
    memory = MemoryBank(str(tmp_path / "memory.sqlite"))
    episode = {
        "episode_id": "ep1",
        "dataset_name": "StrategyQA",
        "sample_id": "0",
        "question": "Q?",
        "golden_answers": ["yes"],
        "answer_stage": {"final_answer": "yes", "stop_reason": "STOP_MAX_ROUNDS"},
        "success_label": "SUCCESS",
    }
    memory.save_episode(episode)
    snapshot_id = memory.save_retrieval_snapshot(
        {
            "episode_id": "ep1",
            "entity_or_topic": "topic",
            "question": "Q?",
            "query_pool": ["Q?"],
            "docs": [{"fingerprint": "abc", "contents": "Doc\nText"}],
            "answer": "yes",
        }
    )
    assert snapshot_id
    assert memory.get_recent_snapshots("topic")
