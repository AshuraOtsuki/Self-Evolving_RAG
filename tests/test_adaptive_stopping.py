from model.adaptive_stopping import AdaptiveStoppingController


def test_high_agreement_and_evidence_stops():
    controller = AdaptiveStoppingController({"min_rounds": 1})
    query_pool = {"q": [{"contents": "Doc\nThe answer is Paris."}]}
    metrics = controller.score_round("Capital?", query_pool, [], "The answer is: Paris", "Paris", "Paris")
    stop, reason = controller.should_stop(metrics, 0)
    assert stop
    assert reason == "STOP_CONFIDENT_AGREEMENT"


def test_min_rounds_continue():
    controller = AdaptiveStoppingController({"min_rounds": 2})
    stop, reason = controller.should_stop(
        {
            "agreement_score": 1.0,
            "evidence_support_score": 1.0,
            "answer_stability_score": 1.0,
            "drift_risk_score": 0.0,
        },
        0,
    )
    assert not stop
    assert reason == "CONTINUE_MIN_ROUNDS"


def test_high_drift_stops():
    controller = AdaptiveStoppingController()
    stop, reason = controller.should_stop(
        {
            "agreement_score": 0.1,
            "evidence_support_score": 0.1,
            "answer_stability_score": 0.1,
            "drift_risk_score": 0.9,
        },
        1,
    )
    assert stop
    assert reason == "STOP_DRIFT_RISK"
