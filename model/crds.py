import hashlib
import re
from datetime import datetime, timezone

from .memory_bank import keyword_overlap, utc_now
from .schemas import ENTITY_TOPIC_SCHEMA


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def fingerprint_doc(title, text):
    raw = f"{title}\n{(text or '')[:500]}"
    return hashlib.sha256(normalize(raw).encode("utf-8")).hexdigest()


class CRDSDetector:
    def __init__(self, memory_bank, generator=None, config: dict | None = None):
        self.memory_bank = memory_bank
        self.generator = generator
        self.config = config or {}

    def build_snapshot(self, episode: dict) -> dict:
        entity = self.extract_entity_topic(episode.get("question", {}))
        docs = []
        query_pool = episode.get("query_stage", {}).get("final_query_pool", {})
        for query, retrieved_docs in query_pool.items():
            for rank, doc in enumerate(retrieved_docs, start=1):
                title, text = self._title_text(doc)
                docs.append(
                    {
                        "doc_id": doc.get("id") or doc.get("doc_id"),
                        "title": title,
                        "text": text,
                        "fingerprint": fingerprint_doc(title, text),
                        "rank": rank,
                        "query": query,
                        "timestamp": doc.get("timestamp"),
                        "source": doc.get("source"),
                    }
                )
        return {
            "snapshot_id": None,
            "episode_id": episode["episode_id"],
            "entity_or_topic": entity.get("entity_or_topic"),
            "question": episode.get("question"),
            "query_pool": list(query_pool.keys()),
            "docs": docs,
            "extracted_facts": self._facts_from_episode(episode),
            "answer": episode.get("answer_stage", {}).get("final_answer"),
            "created_at": utc_now(),
            "topic_meta": entity,
        }

    def detect(self, episode: dict, current_snapshot: dict) -> dict:
        entity = current_snapshot.get("entity_or_topic")
        old_snapshots = []
        if self.memory_bank and entity:
            old_snapshots = [
                snap
                for snap in self.memory_bank.get_recent_snapshots(entity, limit=self.config.get("snapshot_limit", 5))
                if snap.get("episode_id") != episode.get("episode_id")
            ]
        if not old_snapshots:
            return {
                "entity_or_topic": entity,
                "old_snapshots": [],
                "drift_score": 0.0,
                "drift_type": "NO_DRIFT",
                "memory_ops": [],
                "raw_output": None,
            }
        scores = [self._score_pair(old, current_snapshot) for old in old_snapshots]
        best = max(scores, key=lambda item: item["drift_score"])
        ops = self._memory_ops(entity, current_snapshot, best)
        return {
            "entity_or_topic": entity,
            "old_snapshots": old_snapshots,
            "drift_score": best["drift_score"],
            "drift_type": best["drift_type"],
            "memory_ops": ops,
            "raw_output": best,
        }

    def extract_entity_topic(self, question):
        if self.generator and self.config.get("use_llm_entity_extractor", False):
            try:
                return self.generator.generate_json(
                    f"Extract the main entity/topic for retrieval drift tracking.\nQuestion: {question}",
                    ENTITY_TOPIC_SCHEMA,
                    "entity_topic",
                    instructions="Return JSON only.",
                )
            except Exception:
                pass
        words = re.findall(r"[A-Z][A-Za-z0-9'()-]*(?:\s+[A-Z][A-Za-z0-9'()-]*)*", str(question))
        topic = max(words, key=len) if words else " ".join(str(question).split()[:6])
        return {
            "entity_or_topic": topic,
            "main_entity": topic,
            "relation": None,
            "temporal_intent": "current" if re.search(r"\btoday|current|now|recent\b", str(question), re.I) else "unknown",
            "requires_freshness": bool(re.search(r"\btoday|current|now|recent\b", str(question), re.I)),
            "confidence": 0.45,
        }

    def _score_pair(self, old, current):
        old_fps = {doc.get("fingerprint") for doc in old.get("docs", []) if doc.get("fingerprint")}
        new_fps = {doc.get("fingerprint") for doc in current.get("docs", []) if doc.get("fingerprint")}
        overlap = len(old_fps & new_fps) / max(1, len(old_fps | new_fps))
        doc_set_distance = 1 - overlap
        fact_contradiction = self._fact_contradiction(old.get("extracted_facts", []), current.get("extracted_facts", []))
        answer_change = 0.0 if normalize(old.get("answer")) == normalize(current.get("answer")) else 1.0
        timestamp_gap = self._timestamp_gap(old.get("created_at"), current.get("created_at"), current)
        source_change = min(1.0, doc_set_distance)
        drift_score = (
            0.30 * doc_set_distance
            + 0.25 * fact_contradiction
            + 0.20 * answer_change
            + 0.15 * timestamp_gap
            + 0.10 * source_change
        )
        drift_type = "NO_DRIFT"
        if drift_score >= 0.25:
            drift_type = "DOC_SET_DRIFT"
        if fact_contradiction >= 0.8:
            drift_type = "FACT_CONTRADICTION"
        elif answer_change == 1.0 and current.get("topic_meta", {}).get("requires_freshness"):
            drift_type = "ANSWER_CHANGE"
        elif timestamp_gap >= 0.8 and current.get("topic_meta", {}).get("requires_freshness"):
            drift_type = "TEMPORAL_STALENESS"
        elif doc_set_distance >= 0.7:
            drift_type = "DOC_SET_DRIFT"
        return {
            "drift_score": round(drift_score, 4),
            "drift_type": drift_type,
            "doc_set_distance": round(doc_set_distance, 4),
            "fact_contradiction_score": fact_contradiction,
            "answer_change_score": answer_change,
            "timestamp_gap_score": round(timestamp_gap, 4),
            "source_change_score": round(source_change, 4),
        }

    def _memory_ops(self, entity, current_snapshot, score):
        ops = []
        if score["drift_type"] in {"ANSWER_CHANGE", "TEMPORAL_STALENESS", "FACT_CONTRADICTION"}:
            ops.append(
                {
                    "op_type": "ADD",
                    "payload": {
                        "lesson_type": "temporal_retrieval",
                        "target_role": "query_challenger",
                        "trigger_condition": f"query asks about current or time-sensitive status for {entity}",
                        "recommended_action": "add explicit temporal qualifier and prefer fresh retrieved evidence",
                        "entity_or_topic": entity,
                        "confidence": 0.75,
                    },
                    "reason": f"CRDS detected {score['drift_type']}",
                }
            )
        elif score["drift_type"] == "DOC_SET_DRIFT" and score["drift_score"] >= 0.45:
            ops.append(
                {
                    "op_type": "ADD",
                    "payload": {
                        "lesson_type": "retrieval_failure",
                        "target_role": "query_challenger",
                        "trigger_condition": "retrieval for this topic returns low-overlap or weak evidence documents",
                        "recommended_action": "reformulate query using entity + relation + temporal qualifier",
                        "entity_or_topic": entity,
                        "confidence": 0.65,
                    },
                    "reason": "CRDS detected retrieval quality drift",
                }
            )
        return ops

    def _title_text(self, doc):
        contents = doc.get("contents") or doc.get("text") or ""
        if "\n" in contents:
            title, text = contents.split("\n", 1)
        else:
            title, text = doc.get("title") or "Document", contents
        return title, text

    def _facts_from_episode(self, episode):
        facts = []
        for fact in episode.get("metadata", {}).get("facts", []) if isinstance(episode.get("metadata"), dict) else []:
            facts.append(
                {
                    "subject": episode.get("metadata", {}).get("term") or episode.get("question"),
                    "predicate": "fact",
                    "object": fact,
                    "time_scope": None,
                    "supporting_doc_fingerprint": "",
                }
            )
        return facts

    def _fact_contradiction(self, old_facts, new_facts):
        old_map = {(f.get("subject"), f.get("predicate")): normalize(f.get("object")) for f in old_facts}
        for fact in new_facts:
            key = (fact.get("subject"), fact.get("predicate"))
            if key in old_map and old_map[key] != normalize(fact.get("object")):
                return 1.0
        return 0.0

    def _timestamp_gap(self, old_timestamp, current_timestamp, current):
        try:
            old = datetime.fromisoformat(old_timestamp)
            new = datetime.fromisoformat(current_timestamp)
            days = max(0.0, (new - old).total_seconds() / 86400)
        except Exception:
            return 0.0
        if current.get("topic_meta", {}).get("requires_freshness"):
            return min(days / 180, 1.0)
        return min(days / 365, 1.0) * 0.5
