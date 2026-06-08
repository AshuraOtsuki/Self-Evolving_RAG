import json
import re

from .schemas import DTCLS_LESSON_SCHEMA, VALID_LESSON_TYPES, VALID_TARGET_ROLES


GENERIC_TRIGGERS = {
    "when answering questions",
    "when retrieval is bad",
    "when the model is unsure",
    "when asked a question",
}


def normalize_short(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def infer_success_label(episode: dict) -> str:
    gold = episode.get("golden_answers") or []
    pred = episode.get("answer_stage", {}).get("final_answer", "")
    if gold:
        pred_norm = normalize_short(pred)
        if any(normalize_short(answer) == pred_norm for answer in gold):
            return "SUCCESS"
        if pred_norm in {"yes", "no"} and any(normalize_short(answer) in {"yes", "no"} for answer in gold):
            return "FAIL"
    metrics = episode.get("answer_stage", {}).get("stop_metrics", {})
    evidence = float(metrics.get("evidence_support_score", 0.0) or 0.0)
    confidence = 0.0
    rounds = episode.get("answer_stage", {}).get("rounds", [])
    if rounds:
        confidence = float(rounds[-1].get("judge_confidence", 0.0) or 0.0)
    if evidence >= 0.75 and confidence >= 0.75:
        return "SUCCESS"
    if float(metrics.get("drift_risk_score", 0.0) or 0.0) >= 0.75 or evidence < 0.3:
        return "FAIL"
    return "UNKNOWN"


class DTCLSExtractor:
    def __init__(self, generator, config: dict | None = None, memory_bank=None):
        self.generator = generator
        self.config = config or {}
        self.memory_bank = memory_bank

    def extract(self, episode: dict) -> dict:
        if not self.config.get("enabled", True):
            return {"lessons": [], "memory_ops": [], "raw_output": None}
        success_label = infer_success_label(episode)
        query_contrasts = self._query_contrasts(episode, success_label)
        answer_contrasts = self._answer_contrasts(episode, success_label)
        prompt = self._prompt(episode, success_label, query_contrasts, answer_contrasts)
        raw = self.generator.generate_json(
            prompt,
            DTCLS_LESSON_SCHEMA,
            "dtcls_lessons",
            instructions=(
                "Extract reusable, role-specific lessons from a DRAG+SEA debate episode. "
                "Avoid generic advice. Return JSON only."
            ),
            max_output_tokens=self.config.get("max_output_tokens", 900),
        )
        lessons = [lesson for lesson in raw.get("lessons", []) if self._valid_lesson(lesson)]
        ops = [self._lesson_to_op(lesson) for lesson in lessons]
        return {"lessons": lessons, "memory_ops": ops, "raw_output": raw}

    def _query_contrasts(self, episode, success_label):
        contrasts = []
        for round_item in episode.get("query_stage", {}).get("rounds", []):
            contrasts.append(
                {
                    "type": "query_round",
                    "round_idx": round_item.get("round_idx"),
                    "proponent_claim": round_item.get("proponent_argument"),
                    "challenger_claim": round_item.get("challenger_argument"),
                    "judge_decision": round_item.get("judge_decision"),
                    "operation": round_item.get("operation"),
                    "query_pool_before": round_item.get("query_pool_before"),
                    "query_pool_after": round_item.get("query_pool_after"),
                    "outcome": success_label,
                }
            )
        return contrasts

    def _answer_contrasts(self, episode, success_label):
        contrasts = []
        for round_item in episode.get("answer_stage", {}).get("rounds", []):
            contrasts.append(
                {
                    "type": "answer_round",
                    "round_idx": round_item.get("round_idx"),
                    "proponent_answer": round_item.get("proponent_answer"),
                    "challenger_answer": round_item.get("challenger_answer"),
                    "judge_answer": round_item.get("judge_answer"),
                    "agreement_score": round_item.get("agreement_score"),
                    "evidence_support_score": round_item.get("evidence_support_score"),
                    "drift_risk_score": round_item.get("drift_risk_score"),
                    "stop_decision": round_item.get("stop_decision"),
                    "outcome": success_label,
                }
            )
        return contrasts

    def _prompt(self, episode, success_label, query_contrasts, answer_contrasts):
        return f"""Question: {episode.get('question')}
Dataset: {episode.get('dataset_name')}
Gold answers: {episode.get('golden_answers')}
Final answer: {episode.get('answer_stage', {}).get('final_answer')}
Outcome label: {success_label}

Relevant memory used this episode:
{json.dumps(episode.get('relevant_lessons', []), ensure_ascii=False)[:4000]}

Query-stage contrasts:
{json.dumps(query_contrasts, ensure_ascii=False)[:6000]}

Answer-stage contrasts:
{json.dumps(answer_contrasts, ensure_ascii=False)[:6000]}

Task:
Extract 0-5 reusable lessons. Focus on role-specific debate tactics and retrieval/generation failure patterns."""

    def _valid_lesson(self, lesson):
        if float(lesson.get("confidence", 0.0) or 0.0) < self.config.get("min_confidence", 0.55):
            return False
        if lesson.get("lesson_type") not in VALID_LESSON_TYPES:
            return False
        if lesson.get("target_role") not in VALID_TARGET_ROLES:
            return False
        trigger = normalize_short(lesson.get("trigger_condition", ""))
        action = normalize_short(lesson.get("recommended_action", ""))
        if not trigger or not action or trigger in GENERIC_TRIGGERS:
            return False
        if not lesson.get("evidence"):
            return False
        return True

    def _lesson_to_op(self, lesson):
        duplicate = self.memory_bank.find_duplicate(lesson) if self.memory_bank else None
        if duplicate and float(lesson.get("confidence", 0.0)) > float(duplicate.get("confidence", 0.0)):
            return {
                "op_type": "UPDATE",
                "memory_id": duplicate["memory_id"],
                "payload": {
                    "confidence": lesson["confidence"],
                    "metadata_json": json.dumps({"last_evidence": lesson.get("evidence")}),
                },
                "reason": "DTCLS reinforced existing lesson",
            }
        if duplicate:
            return {
                "op_type": "NOOP",
                "memory_id": duplicate["memory_id"],
                "payload": {},
                "reason": "DTCLS duplicate lesson ignored",
            }
        return {
            "op_type": "ADD",
            "memory_id": None,
            "payload": lesson,
            "reason": "DTCLS extracted from episode contrast",
        }
