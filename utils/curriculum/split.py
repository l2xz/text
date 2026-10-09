#!/usr/bin/env python3
# encoding=utf-8
import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def stable_training_row(row: Dict[str, Any], source: str, difficulty: str, margin_key: str) -> Dict[str, Any]:
    keep = (
        "query",
        "prompt",
        "chosen_response",
        "reject_response",
        "chosen",
        "rejected",
    )
    out = {key: row[key] for key in keep if key in row}
    margin = as_float(row.get(margin_key, row.get("client_margin", 0.0)))
    out.update(
        {
            "curriculum_source": source,
            "curriculum_difficulty": difficulty,
            "instruction_margin": margin,
            "client_margin": margin,
            "client_margin_valid": bool(row.get("client_margin_valid", True)),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split scored DPO pairs into easy/medium/hard pools by instruction-model margin."
    )
    parser.add_argument("--input", required=True, help="Scored DPO jsonl, e.g. local_all.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--margin_key", default="client_margin")
    parser.add_argument("--easy_ratio", type=float, default=0.4)
    parser.add_argument("--medium_ratio", type=float, default=0.4)
    parser.add_argument("--source", default="local")
    parser.add_argument("--drop_invalid", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.drop_invalid:
        rows = [row for row in rows if row.get("client_margin_valid", True)]

    # Larger positive instruction margin means the instruction model confidently
    # prefers the chosen response; those are easier/reliable samples.
    rows.sort(key=lambda row: as_float(row.get(args.margin_key, row.get("client_margin", 0.0))), reverse=True)

    n = len(rows)
    n_easy = max(0, min(n, int(round(n * args.easy_ratio))))
    n_medium = max(0, min(n - n_easy, int(round(n * args.medium_ratio))))

    easy = [stable_training_row(row, args.source, "easy", args.margin_key) for row in rows[:n_easy]]
    medium = [
        stable_training_row(row, args.source, "medium", args.margin_key)
        for row in rows[n_easy : n_easy + n_medium]
    ]
    hard = [stable_training_row(row, args.source, "hard", args.margin_key) for row in rows[n_easy + n_medium :]]

    write_jsonl(easy, os.path.join(args.output_dir, "curri_easy.jsonl"))
    write_jsonl(medium, os.path.join(args.output_dir, "curri_medium.jsonl"))
    write_jsonl(hard, os.path.join(args.output_dir, "curri_hard_gap_selected.jsonl"))
    write_jsonl(hard, os.path.join(args.output_dir, "curri_hard_candidates.jsonl"))

    summary = {
        "input": args.input,
        "num_total": n,
        "num_easy": len(easy),
        "num_medium": len(medium),
        "num_hard": len(hard),
        "easy_ratio": args.easy_ratio,
        "medium_ratio": args.medium_ratio,
        "margin_key": args.margin_key,
    }
    with open(os.path.join(args.output_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
