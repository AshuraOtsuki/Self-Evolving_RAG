import re
from collections import Counter


def normalize_answer(text):
    text = (text or "").lower()
    if "the answer is:" in text:
        text = text.split("the answer is:")[-1]
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(text):
    answer = normalize_answer(text)
    if answer in {"yes", "yeah", "true"}:
        return "yes"
    if answer in {"no", "false"}:
        return "no"
    tokens = answer.split()
    if tokens and tokens[-1] in {"yes", "no"}:
        return tokens[-1]
    return answer


def token_f1(a, b):
    a_tokens = normalize_answer(a).split()
    b_tokens = normalize_answer(b).split()
    if not a_tokens or not b_tokens:
        return 0.0
    common = Counter(a_tokens) & Counter(b_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(a_tokens)
    recall = same / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_similarity(a, b):
    left = extract_answer(a)
    right = extract_answer(b)
    if left and right and left == right:
        return 1.0
    return token_f1(left, right)


class AdaptiveStoppingController:
    def __init__(self, config: dict | None = None):
        self.config = {
            "enabled": True,
            "min_rounds": 1,
            "max_rounds": 3,
            "agreement_threshold": 0.85,
            "evidence_threshold": 0.75,
            "stability_threshold": 0.85,
            "drift_risk_threshold": 0.65,
        }
        self.config.update(config or {})

    def score_round(
        self,
        question: str,
        query_pool,
        answer_rounds: list[dict],
        current_proponent_answer: str,
        current_challenger_answer: str,
        current_judge_answer: str,
    ) -> dict:
        answers = [current_proponent_answer, current_challenger_answer, current_judge_answer]
        similarities = [
            answer_similarity(answers[i], answers[j])
            for i in range(len(answers))
            for j in range(i + 1, len(answers))
        ]
        agreement = max(similarities) if similarities else 0.0
        evidence = self._evidence_support(current_judge_answer, query_pool)
        if not answer_rounds:
            stability = 0.5
            previous_evidence = evidence
        else:
            stability = answer_similarity(current_judge_answer, answer_rounds[-1].get("judge_answer", ""))
            previous_evidence = float(answer_rounds[-1].get("evidence_support_score", evidence))
        drift = 0.0
        if evidence < previous_evidence - 0.2:
            drift += 0.3
        if stability < 0.4 and answer_rounds:
            drift += 0.2
        if agreement < 0.3 and len(answer_rounds) >= 2:
            drift += 0.2
        return {
            "agreement_score": round(min(1.0, agreement), 4),
            "evidence_support_score": round(min(1.0, evidence), 4),
            "answer_stability_score": round(min(1.0, stability), 4),
            "drift_risk_score": round(min(1.0, drift), 4),
        }

    def should_stop(self, metrics: dict, round_idx: int) -> tuple[bool, str]:
        if round_idx + 1 < int(self.config["min_rounds"]):
            return False, "CONTINUE_MIN_ROUNDS"
        if metrics["drift_risk_score"] >= float(self.config["drift_risk_threshold"]):
            return True, "STOP_DRIFT_RISK"
        if (
            metrics["agreement_score"] >= float(self.config["agreement_threshold"])
            and metrics["evidence_support_score"] >= float(self.config["evidence_threshold"])
        ):
            return True, "STOP_CONFIDENT_AGREEMENT"
        if (
            metrics["evidence_support_score"] >= float(self.config["evidence_threshold"])
            and metrics["answer_stability_score"] >= float(self.config["stability_threshold"])
        ):
            return True, "STOP_STABLE_GROUNDED"
        if round_idx + 1 >= int(self.config["max_rounds"]):
            return True, "STOP_MAX_ROUNDS"
        return False, "CONTINUE"

    def _evidence_support(self, judge_answer, query_pool):
        answer = extract_answer(judge_answer)
        docs_text = self._query_pool_text(query_pool).lower()
        if not answer:
            return 0.0
        if answer in {"yes", "no"}:
            return 0.8 if docs_text else 0.2
        if answer.lower() in docs_text:
            return 1.0
        f1 = token_f1(answer, docs_text)
        if f1 >= 0.25:
            return 0.7
        return 0.2 if docs_text else 0.0

    def _query_pool_text(self, query_pool):
        chunks = []
        for docs in query_pool.values() if isinstance(query_pool, dict) else []:
            for doc in docs:
                if isinstance(doc, dict):
                    chunks.append(doc.get("contents") or doc.get("text") or "")
        return "\n".join(chunks)
