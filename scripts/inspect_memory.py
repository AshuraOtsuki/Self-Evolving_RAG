import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect DRAG+SEA SQLite memory.")
    parser.add_argument("--memory_path", default="output/drag_sea/memory.sqlite")
    parser.add_argument("--list", default="active", choices=["active", "outdated", "deleted", "all"])
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    path = Path(args.memory_path)
    if not path.exists():
        print(f"No memory DB found at {path}")
        return
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    where = "" if args.list == "all" else "WHERE status = ?"
    params = [] if args.list == "all" else [args.list]
    rows = conn.execute(
        f"""
        SELECT memory_id, lesson_type, target_role, entity_or_topic, confidence,
               status, trigger_condition, recommended_action
        FROM memory_entries
        {where}
        ORDER BY last_updated DESC
        LIMIT ?
        """,
        [*params, args.limit],
    ).fetchall()
    for row in rows:
        print(json.dumps(dict(row), ensure_ascii=False))
    if not rows:
        print("No memory entries.")


if __name__ == "__main__":
    main()
