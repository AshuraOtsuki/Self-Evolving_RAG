import json
import math
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def keyword_overlap(a, b):
    left = tokenize(a)
    right = tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


class MemoryBank:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.init_db()

    def init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_entries (
                memory_id TEXT PRIMARY KEY,
                lesson_type TEXT NOT NULL,
                trigger_condition TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                target_role TEXT NOT NULL,
                entity_or_topic TEXT,
                dataset_name TEXT,
                source_episode_id TEXT,
                confidence REAL NOT NULL DEFAULT 0.5,
                usefulness_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                expires_at TEXT,
                metadata_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episode_logs (
                episode_id TEXT PRIMARY KEY,
                dataset_name TEXT,
                sample_id TEXT,
                question TEXT NOT NULL,
                final_answer TEXT,
                gold_answers_json TEXT,
                success_label TEXT,
                stop_reason TEXT,
                created_at TEXT NOT NULL,
                episode_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                entity_or_topic TEXT,
                question TEXT NOT NULL,
                query_pool_json TEXT NOT NULL,
                docs_json TEXT NOT NULL,
                doc_fingerprints_json TEXT NOT NULL,
                extracted_facts_json TEXT,
                answer TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_operations (
                op_id TEXT PRIMARY KEY,
                episode_id TEXT,
                op_type TEXT NOT NULL,
                memory_id TEXT,
                payload_json TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add_memory(self, entry: dict) -> str:
        duplicate = self.find_duplicate(entry)
        if duplicate:
            return duplicate["memory_id"]
        memory_id = entry.get("memory_id") or f"mem_{uuid.uuid4().hex}"
        now = utc_now()
        metadata = dict(entry.get("metadata") or {})
        for key in ["evidence", "failure_mode", "expected_benefit"]:
            if key in entry:
                metadata[key] = entry[key]
        self.conn.execute(
            """
            INSERT INTO memory_entries (
                memory_id, lesson_type, trigger_condition, recommended_action,
                target_role, entity_or_topic, dataset_name, source_episode_id,
                confidence, usefulness_count, failure_count, status, created_at,
                last_updated, expires_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                entry.get("lesson_type", "query_refinement"),
                entry.get("trigger_condition", ""),
                entry.get("recommended_action", ""),
                entry.get("target_role", "all"),
                entry.get("entity_or_topic"),
                entry.get("dataset_name"),
                entry.get("source_episode_id"),
                float(entry.get("confidence", 0.5) or 0.5),
                int(entry.get("usefulness_count", 0) or 0),
                int(entry.get("failure_count", 0) or 0),
                entry.get("status", "active"),
                entry.get("created_at", now),
                entry.get("last_updated", now),
                entry.get("expires_at"),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return memory_id

    def find_duplicate(self, entry):
        rows = self.conn.execute(
            """
            SELECT * FROM memory_entries
            WHERE status = 'active' AND target_role = ? AND lesson_type = ?
            """,
            (entry.get("target_role", "all"), entry.get("lesson_type", "query_refinement")),
        ).fetchall()
        trigger = entry.get("trigger_condition", "")
        action = entry.get("recommended_action", "")
        for row in rows:
            score = keyword_overlap(trigger + " " + action, row["trigger_condition"] + " " + row["recommended_action"])
            if score >= 0.85:
                return dict(row)
        return None

    def update_memory(self, memory_id: str, patch: dict) -> None:
        allowed = {
            "lesson_type",
            "trigger_condition",
            "recommended_action",
            "target_role",
            "entity_or_topic",
            "dataset_name",
            "confidence",
            "usefulness_count",
            "failure_count",
            "status",
            "expires_at",
            "metadata_json",
        }
        updates = {key: value for key, value in patch.items() if key in allowed}
        updates["last_updated"] = utc_now()
        if not updates:
            return
        sql = ", ".join(f"{key} = ?" for key in updates)
        self.conn.execute(
            f"UPDATE memory_entries SET {sql} WHERE memory_id = ?",
            [*updates.values(), memory_id],
        )
        self.conn.commit()

    def mark_outdated(self, memory_id: str, reason: str) -> None:
        row = self.conn.execute(
            "SELECT metadata_json FROM memory_entries WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        metadata = json.loads(row["metadata_json"] or "{}") if row else {}
        metadata["outdated_reason"] = reason
        self.update_memory(memory_id, {"status": "outdated", "metadata_json": json.dumps(metadata)})

    def delete_memory(self, memory_id: str, soft: bool = True) -> None:
        if soft:
            self.update_memory(memory_id, {"status": "deleted"})
        else:
            self.conn.execute("DELETE FROM memory_entries WHERE memory_id = ?", (memory_id,))
            self.conn.commit()

    def retrieve_lessons(
        self,
        question: str,
        entity_or_topic: str | None = None,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM memory_entries WHERE status = 'active'").fetchall()
        scored = []
        for row in rows:
            entry = dict(row)
            text = f"{entry['trigger_condition']} {entry['recommended_action']}"
            topic_match = 0.0
            if entity_or_topic and entry.get("entity_or_topic"):
                topic_match = keyword_overlap(entity_or_topic, entry["entity_or_topic"])
            confidence = float(entry.get("confidence") or 0.0)
            recency = self._recency_score(entry.get("last_updated"))
            usefulness = self._usefulness_score(entry.get("usefulness_count"), entry.get("failure_count"))
            score = (
                0.35 * keyword_overlap(question, text)
                + 0.25 * topic_match
                + 0.20 * confidence
                + 0.10 * recency
                + 0.10 * usefulness
            )
            if score >= min_score:
                entry["score"] = score
                entry["metadata"] = json.loads(entry.pop("metadata_json") or "{}")
                scored.append(entry)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def save_episode(self, episode: dict) -> None:
        answer_stage = episode.get("answer_stage", {})
        self.conn.execute(
            """
            INSERT OR REPLACE INTO episode_logs (
                episode_id, dataset_name, sample_id, question, final_answer,
                gold_answers_json, success_label, stop_reason, created_at, episode_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode["episode_id"],
                episode.get("dataset_name"),
                str(episode.get("sample_id")) if episode.get("sample_id") is not None else None,
                episode.get("question", ""),
                answer_stage.get("final_answer"),
                json.dumps(episode.get("golden_answers"), ensure_ascii=False),
                episode.get("success_label", "UNKNOWN"),
                answer_stage.get("stop_reason"),
                episode.get("created_at", utc_now()),
                json.dumps(episode, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def save_retrieval_snapshot(self, snapshot: dict) -> str:
        snapshot_id = snapshot.get("snapshot_id") or f"snap_{uuid.uuid4().hex}"
        docs = snapshot.get("docs", [])
        fingerprints = [doc.get("fingerprint") for doc in docs if doc.get("fingerprint")]
        self.conn.execute(
            """
            INSERT OR REPLACE INTO retrieval_snapshots (
                snapshot_id, episode_id, entity_or_topic, question, query_pool_json,
                docs_json, doc_fingerprints_json, extracted_facts_json, answer, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                snapshot.get("episode_id"),
                snapshot.get("entity_or_topic"),
                snapshot.get("question", ""),
                json.dumps(snapshot.get("query_pool", []), ensure_ascii=False),
                json.dumps(docs, ensure_ascii=False),
                json.dumps(fingerprints, ensure_ascii=False),
                json.dumps(snapshot.get("extracted_facts", []), ensure_ascii=False),
                snapshot.get("answer"),
                snapshot.get("created_at", utc_now()),
            ),
        )
        self.conn.commit()
        return snapshot_id

    def get_recent_snapshots(self, entity_or_topic: str, limit: int = 5) -> list[dict]:
        if not entity_or_topic:
            return []
        rows = self.conn.execute(
            """
            SELECT * FROM retrieval_snapshots
            WHERE lower(entity_or_topic) = lower(?)
            ORDER BY created_at DESC LIMIT ?
            """,
            (entity_or_topic, limit),
        ).fetchall()
        return [self._snapshot_from_row(row) for row in rows]

    def apply_memory_ops(self, ops: list[dict], episode_id: str) -> list[dict]:
        applied = []
        for op in ops:
            op_type = op.get("op_type", "NOOP")
            payload = op.get("payload") or {}
            memory_id = op.get("memory_id")
            if op_type == "ADD":
                memory_id = self.add_memory({**payload, "source_episode_id": episode_id})
            elif op_type == "UPDATE" and memory_id:
                self.update_memory(memory_id, payload)
            elif op_type == "DELETE" and memory_id:
                self.delete_memory(memory_id, soft=True)
            elif op_type == "DOWNWEIGHT" and memory_id:
                row = self.conn.execute(
                    "SELECT confidence, failure_count FROM memory_entries WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                if row:
                    self.update_memory(
                        memory_id,
                        {
                            "confidence": max(0.0, float(row["confidence"]) - 0.1),
                            "failure_count": int(row["failure_count"]) + 1,
                        },
                    )
            elif op_type in {"MARK_OUTDATED", "TEMPORAL_WARNING"} and memory_id:
                self.mark_outdated(memory_id, op.get("reason", "Marked by memory operation."))
            record = {**op, "memory_id": memory_id, "episode_id": episode_id}
            self._log_op(record)
            applied.append(record)
        return applied

    def _log_op(self, op):
        self.conn.execute(
            """
            INSERT INTO memory_operations (
                op_id, episode_id, op_type, memory_id, payload_json, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                op.get("op_id") or f"op_{uuid.uuid4().hex}",
                op.get("episode_id"),
                op.get("op_type", "NOOP"),
                op.get("memory_id"),
                json.dumps(op.get("payload") or {}, ensure_ascii=False),
                op.get("reason"),
                utc_now(),
            ),
        )
        self.conn.commit()

    def _snapshot_from_row(self, row):
        data = dict(row)
        data["query_pool"] = json.loads(data.pop("query_pool_json") or "[]")
        data["docs"] = json.loads(data.pop("docs_json") or "[]")
        data["doc_fingerprints"] = json.loads(data.pop("doc_fingerprints_json") or "[]")
        data["extracted_facts"] = json.loads(data.pop("extracted_facts_json") or "[]")
        return data

    def _recency_score(self, timestamp):
        if not timestamp:
            return 0.5
        try:
            dt = datetime.fromisoformat(timestamp)
            age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
            return math.exp(-age_days / 60)
        except Exception:
            return 0.5

    def _usefulness_score(self, useful, failed):
        useful = int(useful or 0)
        failed = int(failed or 0)
        return (useful + 1) / (useful + failed + 2)
