import json
import os

from .memory_bank import MemoryBank


class Mem0MemoryBank(MemoryBank):
    """SQLite-backed DRAG-SEA memory bank with optional mem0 semantic indexing.

    SQLite remains the canonical store for reproducible experiment state. mem0 is
    used as a retrieval index for active lessons and can be disabled or absent
    without changing the MemoryBank contract.
    """

    def __init__(
        self,
        path: str,
        client=None,
        provider: str = "oss",
        api_key: str | None = None,
        user_id: str = "drag_sea",
        agent_id: str = "drag_sea",
        app_id: str | None = None,
        run_id: str | None = None,
        search_threshold: float = 0.0,
        infer: bool = False,
        mem0_config: dict | None = None,
        enabled: bool = True,
    ):
        super().__init__(path)
        self.provider = provider
        self.user_id = user_id
        self.agent_id = agent_id
        self.app_id = app_id
        self.run_id = run_id
        self.search_threshold = search_threshold
        self.infer = infer
        self.mem0 = client if enabled else None
        self._mem0_error = None
        if self.mem0 is None and enabled:
            self.mem0 = self._build_client(provider, api_key, mem0_config)

    def add_memory(self, entry: dict) -> str:
        duplicate = self.find_duplicate(entry)
        if duplicate:
            return duplicate["memory_id"]
        memory_id = super().add_memory(entry)
        row = self._get_memory_row(memory_id)
        if row:
            self._index_lesson(row)
        return memory_id

    def update_memory(self, memory_id: str, patch: dict) -> None:
        super().update_memory(memory_id, patch)
        row = self._get_memory_row(memory_id)
        if row and row.get("status") == "active":
            self._index_lesson(row)

    def retrieve_lessons(
        self,
        question: str,
        entity_or_topic: str | None = None,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> list[dict]:
        if self.mem0 is None:
            return super().retrieve_lessons(question, entity_or_topic, top_k, min_score)

        lessons = []
        seen = set()
        try:
            results = self._search_mem0(question, entity_or_topic, top_k)
            for result in results:
                memory_id = self._result_memory_id(result)
                if not memory_id or memory_id in seen:
                    continue
                row = self._get_memory_row(memory_id)
                if not row or row.get("status") != "active":
                    continue
                score = float(result.get("score", 0.0) or 0.0)
                if score < min_score:
                    continue
                row["score"] = score
                row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
                lessons.append(row)
                seen.add(memory_id)
        except Exception as exc:
            self._mem0_error = str(exc)

        if len(lessons) < top_k:
            fallback = super().retrieve_lessons(
                question,
                entity_or_topic=entity_or_topic,
                top_k=top_k,
                min_score=min_score,
            )
            for lesson in fallback:
                if lesson["memory_id"] not in seen:
                    lessons.append(lesson)
                    seen.add(lesson["memory_id"])
                if len(lessons) >= top_k:
                    break
        lessons.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return lessons[:top_k]

    def _build_client(self, provider, api_key, mem0_config):
        try:
            if provider == "platform":
                from mem0 import MemoryClient

                return MemoryClient(api_key=api_key or os.getenv("MEM0_API_KEY"))
            from mem0 import Memory

            if mem0_config:
                return Memory.from_config(mem0_config)
            return Memory()
        except Exception as exc:
            self._mem0_error = str(exc)
            return None

    def _index_lesson(self, row: dict) -> None:
        if self.mem0 is None:
            return
        text = self._lesson_text(row)
        metadata = {
            "source": "drag_sea_lesson",
            "memory_id": row["memory_id"],
            "lesson_type": row.get("lesson_type"),
            "target_role": row.get("target_role"),
            "entity_or_topic": row.get("entity_or_topic"),
            "dataset_name": row.get("dataset_name"),
            "status": row.get("status"),
        }
        try:
            result = self.mem0.add(
                messages=[{"role": "user", "content": text}],
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.run_id,
                app_id=self.app_id,
                metadata=metadata,
                infer=self.infer,
            )
            self._store_mem0_ids(row["memory_id"], result)
        except TypeError:
            result = self.mem0.add(
                text,
                user_id=self.user_id,
                metadata=metadata,
                infer=self.infer,
            )
            self._store_mem0_ids(row["memory_id"], result)
        except Exception as exc:
            self._mem0_error = str(exc)

    def _search_mem0(self, question, entity_or_topic, top_k):
        query = question if not entity_or_topic else f"{question}\nEntity/topic: {entity_or_topic}"
        kwargs = {
            "filters": self._scope_filters(),
            "top_k": top_k,
            "threshold": self.search_threshold,
        }
        try:
            raw = self.mem0.search(query, **kwargs)
        except TypeError:
            raw = self.mem0.search(query, filters=self._scope_filters())
        if isinstance(raw, dict):
            return raw.get("results", [])
        return raw or []

    def _scope_filters(self):
        filters = {
            "user_id": self.user_id,
            "source": "drag_sea_lesson",
            "status": "active",
        }
        if self.agent_id:
            filters["agent_id"] = self.agent_id
        if self.app_id:
            filters["app_id"] = self.app_id
        if self.run_id:
            filters["run_id"] = self.run_id
        return filters

    def _result_memory_id(self, result):
        metadata = result.get("metadata") or {}
        if metadata.get("memory_id"):
            return metadata["memory_id"]
        return result.get("memory_id")

    def _lesson_text(self, row):
        return (
            f"Type: {row.get('lesson_type')} | Target: {row.get('target_role')} | "
            f"Trigger: {row.get('trigger_condition')} | "
            f"Action: {row.get('recommended_action')} | "
            f"Entity/topic: {row.get('entity_or_topic') or 'general'}"
        )

    def _get_memory_row(self, memory_id):
        row = self.conn.execute(
            "SELECT * FROM memory_entries WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        return dict(row) if row else None

    def _store_mem0_ids(self, memory_id, result):
        ids = self._extract_mem0_ids(result)
        if not ids:
            return
        row = self._get_memory_row(memory_id)
        if not row:
            return
        metadata = json.loads(row.get("metadata_json") or "{}")
        existing = metadata.get("mem0_ids") or []
        metadata["mem0_ids"] = list(dict.fromkeys([*existing, *ids]))
        super().update_memory(memory_id, {"metadata_json": json.dumps(metadata)})

    def _extract_mem0_ids(self, result):
        if isinstance(result, list):
            items = result
        elif isinstance(result, dict):
            items = result.get("results") or result.get("memories") or [result]
        else:
            items = []
        ids = []
        for item in items:
            if isinstance(item, dict):
                value = item.get("id") or item.get("memory_id")
                if value:
                    ids.append(value)
        return ids
