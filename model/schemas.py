QUERY_CHALLENGER_SCHEMA = {
    "type": "object",
    "properties": {
        "argument": {"type": "string"},
        "operation": {"type": "string", "enum": ["KEEP", "QUERY_OPTIMIZATION", "QUERY_EXPANSION"]},
        "original_query": {"type": ["string", "null"]},
        "new_query": {"type": ["string", "null"]},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "expected_improvement": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "argument",
        "operation",
        "original_query",
        "new_query",
        "missing_information",
        "expected_improvement",
        "confidence",
    ],
    "additionalProperties": False,
}

QUERY_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["PROPONENT", "CHALLENGER"]},
        "decision": {"type": "string", "enum": ["STOP", "APPLY_OPERATION"]},
        "reason": {"type": "string"},
        "operation_type": {"type": "string", "enum": ["KEEP", "QUERY_OPTIMIZATION", "QUERY_EXPANSION"]},
        "original_query": {"type": ["string", "null"]},
        "new_query": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "winner",
        "decision",
        "reason",
        "operation_type",
        "original_query",
        "new_query",
        "confidence",
    ],
    "additionalProperties": False,
}

ANSWER_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "normalized_short_answer": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_doc_titles": {"type": "array", "items": {"type": "string"}},
        "detected_conflicts": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "final_answer",
        "normalized_short_answer",
        "reason",
        "confidence",
        "evidence_doc_titles",
        "detected_conflicts",
    ],
    "additionalProperties": False,
}

ENTITY_TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_or_topic": {"type": ["string", "null"]},
        "main_entity": {"type": ["string", "null"]},
        "relation": {"type": ["string", "null"]},
        "temporal_intent": {
            "type": "string",
            "enum": ["current", "historical", "timeless", "unknown"],
        },
        "requires_freshness": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "entity_or_topic",
        "main_entity",
        "relation",
        "temporal_intent",
        "requires_freshness",
        "confidence",
    ],
    "additionalProperties": False,
}

DTCLS_LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lesson_type": {"type": "string"},
                    "target_role": {"type": "string"},
                    "trigger_condition": {"type": "string"},
                    "recommended_action": {"type": "string"},
                    "entity_or_topic": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                    "failure_mode": {"type": ["string", "null"]},
                    "expected_benefit": {"type": "string"},
                },
                "required": [
                    "lesson_type",
                    "target_role",
                    "trigger_condition",
                    "recommended_action",
                    "entity_or_topic",
                    "confidence",
                    "evidence",
                    "failure_mode",
                    "expected_benefit",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lessons"],
    "additionalProperties": False,
}

EVIDENCE_VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "support_score": {"type": "number", "minimum": 0, "maximum": 1},
        "supporting_doc_titles": {"type": "array", "items": {"type": "string"}},
        "contradiction_found": {"type": "boolean"},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": [
        "support_score",
        "supporting_doc_titles",
        "contradiction_found",
        "contradictions",
        "reason",
    ],
    "additionalProperties": False,
}

VALID_LESSON_TYPES = {
    "query_refinement",
    "query_expansion",
    "query_optimization",
    "early_stop",
    "judge_correction",
    "temporal_retrieval",
    "entity_disambiguation",
    "source_conflict",
    "retrieval_failure",
    "answer_grounding",
    "adaptive_stopping",
}

VALID_TARGET_ROLES = {
    "query_proponent",
    "query_challenger",
    "query_judge",
    "answer_proponent",
    "answer_challenger",
    "answer_judge",
    "adaptive_controller",
    "all",
}
