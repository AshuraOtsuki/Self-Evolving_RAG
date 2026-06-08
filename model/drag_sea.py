import json
import uuid
from datetime import datetime, timezone

try:
    from flashrag.pipeline import BasicPipeline
except Exception:
    class BasicPipeline:
        def __init__(self, config=None, prompt_template=None):
            self.config = config or {}
            self.prompt_template = prompt_template

        def evaluate(self, dataset, do_eval=True):
            return dataset

from .adaptive_stopping import AdaptiveStoppingController
from .crds import CRDSDetector
from .dtcls import DTCLSExtractor, infer_success_label
from .lesson_retriever import LessonRetriever
from .memory_bank import utc_now
from .schemas import ANSWER_JUDGE_SCHEMA, QUERY_CHALLENGER_SCHEMA, QUERY_JUDGE_SCHEMA
from prompts.drag_sea_prompts import (
    answer_challenger_prompt,
    answer_judge_prompt,
    answer_proponent_prompt,
    format_lessons_block,
    query_challenger_prompt,
    query_judge_prompt,
    query_proponent_prompt,
)


def get_question(item) -> str:
    if hasattr(item, "question"):
        return item.question
    if isinstance(item, dict):
        for key in ["question", "query", "input", "prompt"]:
            if key in item:
                return item[key]
    raise ValueError("Cannot find question field")


def get_gold_answers(item) -> list[str] | None:
    if hasattr(item, "golden_answers"):
        value = item.golden_answers
        return value if isinstance(value, list) else [str(value)]
    if isinstance(item, dict):
        for key in ["golden_answers", "answers", "answer", "gold", "label"]:
            if key in item:
                value = item[key]
                if isinstance(value, list):
                    return [str(v) for v in value]
                if isinstance(value, bool):
                    return ["yes" if value else "no"]
                return [str(value)]
    return None


def get_item_id(item, default=None):
    if hasattr(item, "id"):
        return item.id
    if isinstance(item, dict):
        return item.get("id", default)
    return default


def update_item_output(item, key, value):
    if hasattr(item, "update_output"):
        item.update_output(key, value)
    elif isinstance(item, dict):
        item[key] = value


class DRAGSEAPipeline(BasicPipeline):
    def __init__(
        self,
        config,
        generator,
        retriever,
        memory_bank=None,
        max_query_debate_rounds=3,
        max_answer_debate_rounds=3,
        adaptive_config=None,
        memory_config=None,
        signals_config=None,
        save_prompts=False,
        dataset_name=None,
    ):
        if hasattr(config, "__contains__") and "device" in config:
            super().__init__(config)
        else:
            self.config = config or {}
            self.prompt_template = None
        self.config = config
        self.generator = generator
        self.retriever = retriever
        self.memory_bank = memory_bank
        self.max_query_debate_rounds = max_query_debate_rounds
        self.max_answer_debate_rounds = max_answer_debate_rounds
        self.memory_config = {
            "enabled": True,
            "top_k_lessons": 5,
            "min_lesson_score": 0.15,
        }
        self.memory_config.update(memory_config or {})
        self.signals_config = {
            "dtcls_enabled": True,
            "crds_enabled": True,
            "apply_memory_ops": True,
        }
        self.signals_config.update(signals_config or {})
        self.adaptive = AdaptiveStoppingController(
            {**(adaptive_config or {}), "max_rounds": max_answer_debate_rounds}
        )
        self.lesson_retriever = LessonRetriever(
            memory_bank,
            top_k=self.memory_config["top_k_lessons"],
            min_score=self.memory_config["min_lesson_score"],
        )
        self.dtcls = DTCLSExtractor(
            generator,
            {"enabled": self.signals_config.get("dtcls_enabled", True)},
            memory_bank=memory_bank,
        )
        self.crds = CRDSDetector(
            memory_bank,
            generator=generator,
            config={"enabled": self.signals_config.get("crds_enabled", True)},
        )
        self.save_prompts = save_prompts
        self.dataset_name = dataset_name or self._cfg_get("dataset_name")

    def run(self, dataset, do_eval: bool = True):
        episodes = []
        for idx, item in enumerate(dataset):
            episodes.append(self.run_one(item, sample_id=idx, do_eval=False))
        return episodes

    def run_one(self, item, sample_id=None, do_eval: bool = False) -> dict:
        question = get_question(item)
        episode = self._new_episode(item, question, sample_id)
        start_calls = len(getattr(self.generator, "call_records", []))
        relevant_lessons = []
        if self.memory_bank and self.memory_config.get("enabled", True):
            topic = self.crds.extract_entity_topic(question).get("entity_or_topic")
            relevant_lessons = self.lesson_retriever.retrieve(question, topic)
        episode["relevant_lessons"] = relevant_lessons
        query_stage = self.query_stage_debate(item, relevant_lessons)
        episode["query_stage"] = query_stage
        answer_stage = self.answer_stage_debate(item, query_stage["final_query_pool"], relevant_lessons)
        episode["answer_stage"] = answer_stage
        episode["usage"]["llm_calls"] = len(getattr(self.generator, "call_records", [])) - start_calls
        episode["usage"]["retriever_calls"] = sum(
            len(round_item.get("query_pool_after", [])) for round_item in query_stage.get("rounds", [])
        ) or len(query_stage["final_query_pool"])
        episode["success_label"] = infer_success_label(episode)
        self.post_episode_update(episode)
        update_item_output(item, "pred", answer_stage.get("final_answer"))
        update_item_output(item, "DRAG_SEA_Episode", episode)
        return episode

    def run_question(self, question: str) -> dict:
        return self.run_one({"question": question, "id": "question"}, sample_id="question", do_eval=False)

    def query_stage_debate(self, item, relevant_lessons: list[dict]) -> dict:
        question = get_question(item)
        lessons_block = format_lessons_block(relevant_lessons)
        query_pool = {question.strip(): self._retrieve(question)}
        rounds = []
        for round_idx in range(self.max_query_debate_rounds):
            before_queries = list(query_pool.keys())
            docs_before = self._all_docs(query_pool)
            query_pool_text = self.format_query_pool(query_pool)
            prop_prompt = query_proponent_prompt(question, query_pool_text, lessons_block)
            proponent = self.generator.generate(prop_prompt)[0]
            challenger_prompt_text = query_challenger_prompt(question, query_pool_text, lessons_block)
            challenger = self.generator.generate_json(
                challenger_prompt_text,
                QUERY_CHALLENGER_SCHEMA,
                "query_challenger",
                instructions="Return JSON only.",
            )
            judge_prompt_text = query_judge_prompt(
                question,
                query_pool_text,
                proponent,
                json.dumps(challenger, ensure_ascii=False),
                lessons_block,
            )
            judge = self.generator.generate_json(
                judge_prompt_text,
                QUERY_JUDGE_SCHEMA,
                "query_judge",
                instructions="Return JSON only.",
            )
            operation = judge.get("operation_type") or challenger.get("operation") or "KEEP"
            if judge.get("decision") == "APPLY_OPERATION" and operation != "KEEP":
                self._apply_query_operation(query_pool, operation, judge, challenger)
            stop = judge.get("decision") == "STOP" or judge.get("winner") == "PROPONENT"
            round_record = {
                "round_idx": round_idx,
                "query_pool_before": before_queries,
                "retrieved_docs_before": docs_before,
                "proponent_argument": proponent,
                "challenger_argument": challenger.get("argument", ""),
                "challenger_output": challenger,
                "judge_decision": judge.get("winner"),
                "judge_reason": judge.get("reason"),
                "operation": "STOP" if stop else operation,
                "operation_payload": {
                    "original_query": judge.get("original_query") or challenger.get("original_query"),
                    "new_query": judge.get("new_query") or challenger.get("new_query"),
                },
                "query_pool_after": list(query_pool.keys()),
            }
            if self.save_prompts:
                round_record["prompts"] = {
                    "proponent": prop_prompt,
                    "challenger": challenger_prompt_text,
                    "judge": judge_prompt_text,
                }
            rounds.append(round_record)
            if stop:
                break
        return {"rounds": rounds, "final_query_pool": query_pool}

    def answer_stage_debate(self, item, query_pool: dict, relevant_lessons: list[dict]) -> dict:
        question = get_question(item)
        lessons_block = format_lessons_block(relevant_lessons)
        rounds = []
        final_answer = ""
        stop_reason = "STOP_MAX_ROUNDS"
        stop_metrics = {}
        strategyqa = self._is_strategyqa()
        for round_idx in range(self.max_answer_debate_rounds):
            history = self._answer_history(rounds)
            query_pool_text = self.format_query_pool(query_pool)
            prop_prompt = answer_proponent_prompt(question, query_pool_text, history, lessons_block, strategyqa)
            proponent = self.generator.generate(prop_prompt)[0]
            challenger_prompt_text = answer_challenger_prompt(
                question,
                f"{history}\nProponent answer:\n{proponent}",
                lessons_block,
                strategyqa,
            )
            challenger = self.generator.generate(challenger_prompt_text)[0]
            judge_prompt_text = answer_judge_prompt(
                question,
                query_pool_text,
                proponent,
                challenger,
                strategyqa,
            )
            judge = self.generator.generate_json(
                judge_prompt_text,
                ANSWER_JUDGE_SCHEMA,
                "answer_judge",
                instructions="Return JSON only.",
            )
            judge_answer = judge.get("normalized_short_answer") or judge.get("final_answer", "")
            metrics = self.adaptive.score_round(
                question,
                query_pool,
                rounds,
                proponent,
                challenger,
                judge_answer,
            )
            should_stop, reason = self.adaptive.should_stop(metrics, round_idx)
            round_record = {
                "round_idx": round_idx,
                "proponent_answer": proponent,
                "challenger_answer": challenger,
                "judge_answer": judge_answer,
                "judge_reason": judge.get("reason"),
                "judge_confidence": judge.get("confidence", 0.0),
                **metrics,
                "stop_decision": reason,
                "judge_output": judge,
            }
            if self.save_prompts:
                round_record["prompts"] = {
                    "proponent": prop_prompt,
                    "challenger": challenger_prompt_text,
                    "judge": judge_prompt_text,
                }
            rounds.append(round_record)
            final_answer = judge_answer
            stop_reason = reason
            stop_metrics = metrics
            if should_stop:
                break
        return {
            "rounds": rounds,
            "final_answer": final_answer,
            "stop_reason": stop_reason,
            "stop_metrics": stop_metrics,
        }

    def post_episode_update(self, episode: dict) -> dict:
        snapshot = self.crds.build_snapshot(episode)
        episode["retrieval_snapshot"] = snapshot
        crds_result = (
            self.crds.detect(episode, snapshot)
            if self.signals_config.get("crds_enabled", True)
            else {"drift_score": None, "memory_ops": [], "raw_output": None}
        )
        dtcls_result = (
            self.dtcls.extract(episode)
            if self.signals_config.get("dtcls_enabled", True)
            else {"lessons": [], "memory_ops": [], "raw_output": None}
        )
        ops = [*dtcls_result.get("memory_ops", []), *crds_result.get("memory_ops", [])]
        applied_ops = []
        if self.memory_bank:
            if self.signals_config.get("apply_memory_ops", True):
                applied_ops = self.memory_bank.apply_memory_ops(ops, episode["episode_id"])
            self.memory_bank.save_retrieval_snapshot(snapshot)
        episode["dtcls"] = {
            "lessons": dtcls_result.get("lessons", []),
            "raw_output": dtcls_result.get("raw_output"),
        }
        episode["crds"] = crds_result
        episode["memory_ops"] = applied_ops
        if self.memory_bank:
            self.memory_bank.save_episode(episode)
        return episode

    def format_query_pool(self, query_pool):
        chunks = []
        for idx, (query, docs) in enumerate(query_pool.items(), start=1):
            chunks.append(f"Query {idx}: {query}")
            chunks.append("Retrieved Content:")
            for doc_idx, doc in enumerate(docs, start=1):
                title, text = self._title_text(doc)
                chunks.append(f"Doc {doc_idx}(Title: {title}) {text}")
        return "\n".join(chunks)

    def _apply_query_operation(self, query_pool, operation, judge, challenger):
        original = judge.get("original_query") or challenger.get("original_query")
        new_query = judge.get("new_query") or challenger.get("new_query")
        if not new_query:
            return
        if operation == "QUERY_OPTIMIZATION" and original:
            key = self._find_query_key(query_pool, original)
            if key:
                query_pool.pop(key, None)
        if new_query not in query_pool:
            query_pool[new_query.strip()] = self._retrieve(new_query)

    def _retrieve(self, query):
        try:
            docs = self.retriever.search(query)
        except Exception:
            docs = []
        return [self._normalize_doc(doc) for doc in docs]

    def _normalize_doc(self, doc):
        if isinstance(doc, dict):
            if "contents" in doc:
                return doc
            title = doc.get("title") or doc.get("id") or "Document"
            text = doc.get("text") or doc.get("contents") or ""
            return {**doc, "contents": f"{title}\n{text}"}
        return {"contents": f"Document\n{str(doc)}"}

    def _find_query_key(self, query_pool, target):
        target = (target or "").lower()
        best_key = None
        best_score = -1
        for key in query_pool:
            score = len(set(key.lower().split()) & set(target.split()))
            if score > best_score:
                best_key = key
                best_score = score
        return best_key

    def _all_docs(self, query_pool):
        docs = []
        for query, results in query_pool.items():
            for doc in results:
                docs.append({"query": query, **doc})
        return docs

    def _answer_history(self, rounds):
        if not rounds:
            return "No previous answer debate rounds."
        lines = []
        for item in rounds:
            lines.append(
                "Round {round_idx}: proponent={proponent_answer}; challenger={challenger_answer}; "
                "judge={judge_answer}".format(**item)
            )
        return "\n".join(lines)

    def _title_text(self, doc):
        contents = doc.get("contents") or ""
        if "\n" in contents:
            title, text = contents.split("\n", 1)
        else:
            title = doc.get("title") or "Document"
            text = contents
        return title, text

    def _new_episode(self, item, question, sample_id):
        metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})
        return {
            "episode_id": f"ep_{uuid.uuid4().hex}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": self.dataset_name,
            "sample_id": get_item_id(item, sample_id),
            "question": question,
            "golden_answers": get_gold_answers(item),
            "metadata": metadata,
            "relevant_lessons": [],
            "query_stage": {"rounds": [], "final_query_pool": {}},
            "answer_stage": {"rounds": [], "final_answer": "", "stop_reason": "", "stop_metrics": {}},
            "retrieval_snapshot": {},
            "dtcls": {"lessons": [], "raw_output": None},
            "crds": {"drift_score": None, "memory_ops": [], "raw_output": None},
            "usage": {
                "llm_calls": 0,
                "retriever_calls": 0,
                "estimated_input_tokens": None,
                "estimated_output_tokens": None,
            },
        }

    def _cfg_get(self, key, default=None):
        try:
            return self.config[key]
        except Exception:
            return default

    def _is_strategyqa(self):
        return str(self.dataset_name or "").lower() == "strategyqa"
