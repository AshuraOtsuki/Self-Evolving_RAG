from model.dtcls import DTCLSExtractor, infer_success_label


class FakeGenerator:
    def generate_json(self, *args, **kwargs):
        return {
            "lessons": [
                {
                    "lesson_type": "query_expansion",
                    "target_role": "query_challenger",
                    "trigger_condition": "question requires comparing two named entities",
                    "recommended_action": "add one focused query for each entity before judging sufficiency",
                    "entity_or_topic": None,
                    "confidence": 0.8,
                    "evidence": "challenger expansion improved coverage",
                    "failure_mode": None,
                    "expected_benefit": "better multi-hop coverage",
                }
            ]
        }


def test_extracts_add_op():
    episode = {
        "question": "Compare A and B?",
        "dataset_name": "StrategyQA",
        "golden_answers": ["yes"],
        "relevant_lessons": [],
        "query_stage": {
            "rounds": [
                {
                    "round_idx": 0,
                    "proponent_argument": "enough",
                    "challenger_argument": "need B",
                    "judge_decision": "CHALLENGER",
                    "operation": "QUERY_EXPANSION",
                    "query_pool_before": ["A"],
                    "query_pool_after": ["A", "B"],
                }
            ]
        },
        "answer_stage": {
            "final_answer": "yes",
            "rounds": [{"judge_confidence": 0.9}],
            "stop_metrics": {"evidence_support_score": 0.9},
        },
    }
    result = DTCLSExtractor(FakeGenerator(), {}).extract(episode)
    assert result["memory_ops"][0]["op_type"] == "ADD"
    assert infer_success_label(episode) == "SUCCESS"
