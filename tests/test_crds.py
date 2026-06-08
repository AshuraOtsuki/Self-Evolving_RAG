from model.crds import CRDSDetector
from model.memory_bank import MemoryBank


def test_answer_change_for_fresh_topic_creates_warning(tmp_path):
    memory = MemoryBank(str(tmp_path / "memory.sqlite"))
    old = {
        "episode_id": "old",
        "entity_or_topic": "OpenAI CEO",
        "question": "Who is the current OpenAI CEO?",
        "query_pool": ["q"],
        "docs": [{"fingerprint": "old"}],
        "answer": "Alice",
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    memory.save_retrieval_snapshot(old)
    detector = CRDSDetector(memory)
    current = {
        "episode_id": "new",
        "entity_or_topic": "OpenAI CEO",
        "question": "Who is the current OpenAI CEO?",
        "query_pool": ["q"],
        "docs": [{"fingerprint": "new"}],
        "extracted_facts": [],
        "answer": "Bob",
        "created_at": "2025-01-01T00:00:00+00:00",
        "topic_meta": {"requires_freshness": True},
    }
    result = detector.detect({"episode_id": "new"}, current)
    assert result["drift_type"] in {"ANSWER_CHANGE", "TEMPORAL_STALENESS", "DOC_SET_DRIFT"}
    assert result["memory_ops"]
