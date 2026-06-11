# DRAG+SEA JSON Schemas

These schemas are implementation-ready drafts. Put them in `model/schemas.py` as Python dicts.

## Query Challenger Schema

```python
QUERY_CHALLENGER_SCHEMA = {
    "type": "object",
    "properties": {
        "argument": {"type": "string"},
        "operation": {"type": "string", "enum": ["KEEP", "QUERY_OPTIMIZATION", "QUERY_EXPANSION"]},
        "original_query": {"type": ["string", "null"]},
        "new_query": {"type": ["string", "null"]},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "expected_improvement": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["argument", "operation", "original_query", "new_query", "missing_information", "expected_improvement", "confidence"],
    "additionalProperties": False
}
```

## Query Judge Schema

```python
QUERY_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["PROPONENT", "CHALLENGER"]},
        "decision": {"type": "string", "enum": ["STOP", "APPLY_OPERATION"]},
        "reason": {"type": "string"},
        "operation_type": {"type": "string", "enum": ["KEEP", "QUERY_OPTIMIZATION", "QUERY_EXPANSION"]},
        "original_query": {"type": ["string", "null"]},
        "new_query": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["winner", "decision", "reason", "operation_type", "original_query", "new_query", "confidence"],
    "additionalProperties": False
}
```

## Answer Judge Schema

```python
ANSWER_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "normalized_short_answer": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_doc_titles": {"type": "array", "items": {"type": "string"}},
        "detected_conflicts": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["final_answer", "normalized_short_answer", "reason", "confidence", "evidence_doc_titles", "detected_conflicts"],
    "additionalProperties": False
}
```

## Entity Topic Schema

```python
ENTITY_TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_or_topic": {"type": ["string", "null"]},
        "main_entity": {"type": ["string", "null"]},
        "relation": {"type": ["string", "null"]},
        "temporal_intent": {"type": "string", "enum": ["current", "historical", "timeless", "unknown"]},
        "requires_freshness": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["entity_or_topic", "main_entity", "relation", "temporal_intent", "requires_freshness", "confidence"],
    "additionalProperties": False
}
```

## Evidence Verifier Schema

```python
EVIDENCE_VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "support_score": {"type": "number", "minimum": 0, "maximum": 1},
        "supporting_doc_titles": {"type": "array", "items": {"type": "string"}},
        "contradiction_found": {"type": "boolean"},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"}
    },
    "required": ["support_score", "supporting_doc_titles", "contradiction_found", "contradictions", "reason"],
    "additionalProperties": False
}
```
