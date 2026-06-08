def format_lessons_block(lessons):
    if not lessons:
        return "No relevant past lessons."
    lines = ["Relevant past lessons for this episode:"]
    for idx, lesson in enumerate(lessons, start=1):
        lines.append(
            "[{idx}] Type: {lesson_type} | Target: {target_role} | "
            "Trigger: {trigger_condition} | Action: {recommended_action} | "
            "Confidence: {confidence:.2f}".format(
                idx=idx,
                lesson_type=lesson.get("lesson_type", "unknown"),
                target_role=lesson.get("target_role", "all"),
                trigger_condition=lesson.get("trigger_condition", ""),
                recommended_action=lesson.get("recommended_action", ""),
                confidence=float(lesson.get("confidence", 0.0) or 0.0),
            )
        )
    lines.append("Use lessons as tactical guidance, not factual evidence.")
    lines.append("Ignore lessons that conflict with retrieved evidence.")
    return "\n".join(lines)


def query_proponent_prompt(question, query_pool, lessons_block):
    return f"""You are the Query Proponent in a DRAG retrieval debate.
Argue that the current query pool and retrieved documents are sufficient.
Do not propose new queries.

{lessons_block}

Question:
{question}

Current query pool and retrieved documents:
{query_pool}

Return a concise argument explaining why retrieval is sufficient."""


def query_challenger_prompt(question, query_pool, lessons_block):
    return f"""You are the Query Challenger in a DRAG retrieval debate.
Challenge retrieval sufficiency and propose one operation if needed.

Allowed operations:
- KEEP: no change needed
- QUERY_OPTIMIZATION: replace a weak query with a better query
- QUERY_EXPANSION: add a new short query to cover missing information

{lessons_block}

Question:
{question}

Current query pool and retrieved documents:
{query_pool}

Return only JSON matching the requested schema."""


def query_judge_prompt(question, query_pool, proponent_argument, challenger_output, lessons_block):
    return f"""You are the Query Judge.
Decide whether Proponent or Challenger made the stronger argument.
Prefer stopping when documents directly answer the question with low conflict.
Prefer Challenger when query lacks entity disambiguation, temporal qualifier, or multi-hop evidence.
Use lessons as tactical guidance, not facts.

{lessons_block}

Question:
{question}

Current query pool and retrieved documents:
{query_pool}

Proponent argument:
{proponent_argument}

Challenger structured output:
{challenger_output}

Return only JSON matching the requested schema."""


def answer_proponent_prompt(question, query_pool, answer_history, lessons_block, strategyqa=False):
    candidate_rule = "For StrategyQA, choose only Yes or No." if strategyqa else ""
    return f"""You are the Answer Proponent.
Answer the question using retrieved documents as primary evidence.
If evidence is insufficient, state uncertainty. {candidate_rule}
Always end with: The answer is: <short answer>

{lessons_block}

Question:
{question}

Retrieved documents:
{query_pool}

Previous debate context:
{answer_history}"""


def answer_challenger_prompt(question, answer_history, lessons_block, strategyqa=False):
    candidate_rule = "For StrategyQA, choose only Yes or No." if strategyqa else ""
    return f"""You are the Answer Challenger.
Critically examine the Proponent answer. {candidate_rule}
You may use your own knowledge and reasoning, but do not override strong retrieved evidence without explaining the conflict.
Identify possible hallucination, retrieval noise, ambiguity, or missing temporal context.
Always end with: The answer is: <short answer>

{lessons_block}

Question:
{question}

Proponent/current debate context:
{answer_history}"""


def answer_judge_prompt(question, query_pool, proponent_answer, challenger_answer, strategyqa=False):
    candidate_rule = "For StrategyQA, normalized_short_answer must be yes or no." if strategyqa else ""
    return f"""You are the Answer Judge.
Select the best final answer based on retrieved evidence, Proponent argument, Challenger critique, and consistency.
If retrieved evidence is strong, prefer it. {candidate_rule}

Question:
{question}

Retrieved documents:
{query_pool}

Proponent answer:
{proponent_answer}

Challenger answer:
{challenger_answer}

Return only JSON matching the requested schema."""
