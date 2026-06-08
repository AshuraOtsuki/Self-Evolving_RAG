import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export a DRAG+SEA episode JSONL row to markdown.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--sample_id", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    episodes_path = run_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found in {run_dir}")
    selected = None
    with open(episodes_path, "r", encoding="utf-8") as file:
        for line in file:
            episode = json.loads(line)
            if args.sample_id is None or str(episode.get("sample_id")) == str(args.sample_id):
                selected = episode
                break
    if selected is None:
        raise ValueError(f"No episode found for sample_id={args.sample_id}")
    out_path = run_dir / f"episode_{selected.get('sample_id')}_report.md"
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(f"# DRAG+SEA Episode {selected.get('sample_id')}\n\n")
        file.write(f"## Question\n{selected.get('question')}\n\n")
        file.write(f"## Gold / Prediction\nGold: {selected.get('golden_answers')}\n\n")
        file.write(f"Prediction: {selected.get('answer_stage', {}).get('final_answer')}\n\n")
        file.write("## Retrieved Lessons\n")
        for lesson in selected.get("relevant_lessons", []):
            file.write(f"- {lesson.get('lesson_type')}: {lesson.get('recommended_action')}\n")
        file.write("\n## Query Debate\n")
        for round_item in selected.get("query_stage", {}).get("rounds", []):
            file.write(
                f"- Round {round_item.get('round_idx')}: {round_item.get('judge_decision')} / "
                f"{round_item.get('operation')} — {round_item.get('judge_reason')}\n"
            )
        file.write("\n## Answer Debate\n")
        for round_item in selected.get("answer_stage", {}).get("rounds", []):
            file.write(
                f"- Round {round_item.get('round_idx')}: answer={round_item.get('judge_answer')}, "
                f"stop={round_item.get('stop_decision')}, evidence={round_item.get('evidence_support_score')}\n"
            )
        file.write(f"\n## Stop Metrics\n{json.dumps(selected.get('answer_stage', {}).get('stop_metrics'), indent=2)}\n")
        file.write(f"\n## DTCLS Lessons\n{json.dumps(selected.get('dtcls'), ensure_ascii=False, indent=2)}\n")
        file.write(f"\n## CRDS\n{json.dumps(selected.get('crds'), ensure_ascii=False, indent=2)}\n")
    print(out_path)


if __name__ == "__main__":
    main()
