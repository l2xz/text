# encoding=utf-8
import argparse
import json
import math
import os
import random
from collections import Counter


MARGIN_KEYS = [
    "client_margin",
    "dpo_margin",
    "reward_gap",
    "implicit_reward_gap",
    "teacher_margin",
    "margin",
    "margins",
]


DEFAULT_BEES_MARGIN_KEYS = [
    "client_margin",
    "reward_gap",
    "implicit_reward_gap",
    "teacher_margin",
    "dpo_margin",
    "margin",
    "margins",
]


def read_jsonl(path):
    rows = []
    bad = 0
    if not os.path.exists(path):
        return rows, bad
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                bad += 1
                print(f"[Warning] Bad JSON skipped: {path}, line={line_id}, error={exc}")
    return rows, bad


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_source(spec):
    if "=" not in spec:
        raise ValueError(f"Invalid --source '{spec}', expected name=/path/file.jsonl")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid --source '{spec}', empty source name")
    return name, path


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def get_margin(row):
    if row.get("client_margin_valid") is False:
        return None
    for key in MARGIN_KEYS:
        if key in row:
            value = safe_float(row.get(key), None)
            if value is not None:
                return value
    return None


def parse_csv_list(value):
    return [part.strip() for part in str(value).split(",") if part.strip()]


def get_margin_sources(row, keys):
    if row.get("client_margin_valid") is False:
        return {}
    values = {}
    for key in keys:
        value = safe_float(row.get(key), None)
        if value is not None:
            values[key] = value
    return values


def make_key(row):
    return json.dumps(
        {
            "query": row.get("query", row.get("prompt", "")),
            "chosen": row.get("chosen_response", row.get("chosen", "")),
            "rejected": row.get("reject_response", row.get("rejected", "")),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def is_public(source):
    return str(source).startswith("public_")


def difficulty(source):
    source = str(source)
    if "easy" in source:
        return "easy"
    if "medium" in source:
        return "medium"
    if "hard" in source:
        return "hard"
    return "unknown"


def margin_to_probability(margin, low, high, floor):
    if high <= low:
        raise ValueError("--bees_margin_high must be larger than --bees_margin_low")
    clipped = min(max(margin, low), high)
    prob = (clipped - low) / (high - low)
    floor = min(max(floor, 0.0), 0.49)
    return min(max(prob, floor), 1.0 - floor)


def bayesian_aggregate(probs):
    if not probs:
        return None
    pos = 1.0
    neg = 1.0
    for prob in probs:
        pos *= prob
        neg *= 1.0 - prob
    denom = pos + neg
    if denom <= 0:
        return None
    return pos / denom


def bees_probability(margin_sources, args):
    probs = [
        margin_to_probability(value, args.bees_margin_low, args.bees_margin_high, args.bees_prob_floor)
        for value in margin_sources.values()
    ]
    return bayesian_aggregate(probs)


def score_margin(margin, tau, sigma, risk_lambda, instability, score_mode, bees_prob, bees_min_prob):
    learnability = math.exp(-abs(margin) / max(tau, 1e-8))
    if score_mode == "boundary":
        return learnability
    if score_mode == "bees":
        return bees_prob if bees_prob is not None else float("-inf")
    if score_mode == "reliability_boundary":
        if bees_prob is None:
            return float("-inf")
        reliability = max(0.0, bees_prob - bees_min_prob) / max(1.0 - bees_min_prob, 1e-8)
        return learnability * reliability
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    q = min(max(q, 0.0), 1.0)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def prepare_rows(sources, args, instability):
    prepared = []
    bad_lines = 0
    bees_margin_keys = parse_csv_list(args.bees_margin_keys)
    for spec in args.source:
        source, path = parse_source(spec)
        rows, bad = read_jsonl(path)
        bad_lines += bad
        for row in rows:
            margin = get_margin(row)
            if margin is None:
                continue
            margin_sources = get_margin_sources(row, bees_margin_keys)
            if len(margin_sources) < args.bees_min_sources:
                continue
            bees_prob = bees_probability(margin_sources, args)
            if args.min_margin is not None and margin < args.min_margin:
                continue
            if args.drop_truncated and row.get("client_truncated") is True:
                continue
            item = dict(row)
            item["_model_select_source"] = source
            item["_model_select_origin"] = "public" if is_public(source) else "local"
            item["_model_select_difficulty"] = difficulty(source)
            item["_model_select_margin"] = margin
            item["_model_select_margin_sources"] = margin_sources
            item["_model_select_bees_prob"] = bees_prob
            item["_model_select_bees_source_count"] = len(margin_sources)
            item["_model_select_score"] = score_margin(
                margin,
                args.tau,
                args.sigma,
                args.risk_lambda,
                instability,
                args.score_mode,
                bees_prob,
                args.bees_min_prob,
            )
            prepared.append(item)
    return prepared, bad_lines


def apply_hard_gate(rows, args):
    if not args.auto_gate_hard:
        return rows, None, 0

    local_non_hard_scores = [
        row["_model_select_score"]
        for row in rows
        if row["_model_select_origin"] == "local" and row["_model_select_difficulty"] != "hard"
    ]
    threshold = quantile(local_non_hard_scores, args.hard_threshold_quantile)
    if threshold is None:
        return rows, None, 0

    kept = []
    dropped = 0
    for row in rows:
        if row["_model_select_difficulty"] == "hard" and row["_model_select_score"] < threshold:
            dropped += 1
            continue
        kept.append(row)
    return kept, threshold, dropped


def dedup_preserve_order(rows):
    out = []
    seen = set()
    for row in rows:
        key = make_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def count_by(rows, key):
    return dict(sorted(Counter(row.get(key, "unknown") for row in rows).items()))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def training_row(row):
    """Keep only stable DPO training fields for datasets.load_dataset(json)."""
    preferred_keys = (
        "query",
        "prompt",
        "chosen_response",
        "reject_response",
        "chosen",
        "rejected",
    )
    return {key: row[key] for key in preferred_keys if key in row}


def main():
    parser = argparse.ArgumentParser(
        description="Model-based easy/medium curriculum allocation with local-data guarantee."
    )
    parser.add_argument("--source", action="append", required=True, help="name=/path/file.jsonl")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target_size", type=int, required=True)
    parser.add_argument("--train_stats", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--risk_lambda", type=float, default=1.0)
    parser.add_argument(
        "--score_mode",
        choices=("boundary", "bees", "reliability_boundary"),
        default="bees",
        help=(
            "boundary scores samples by closeness to the DPO decision boundary exp(-|m|/tau); "
            "bees selects high-confidence preference pairs via BeeS-style probability aggregation; "
            "reliability_boundary keeps boundary pressure but gates ambiguous/noisy pairs by BeeS confidence."
        ),
    )
    parser.add_argument(
        "--bees_margin_keys",
        type=str,
        default=",".join(DEFAULT_BEES_MARGIN_KEYS),
        help="Comma-separated margin fields to aggregate for BeeS-style confidence.",
    )
    parser.add_argument(
        "--bees_margin_low",
        type=float,
        default=-0.5,
        help="Lower margin bound L for projecting each source into preference probability.",
    )
    parser.add_argument(
        "--bees_margin_high",
        type=float,
        default=0.5,
        help="Upper margin bound U for projecting each source into preference probability.",
    )
    parser.add_argument(
        "--bees_min_prob",
        type=float,
        default=0.55,
        help="Reliability floor used by reliability_boundary; values below it receive zero boundary score.",
    )
    parser.add_argument(
        "--bees_prob_floor",
        type=float,
        default=1e-4,
        help="Numerical floor/ceiling for per-source preference probabilities.",
    )
    parser.add_argument(
        "--bees_min_sources",
        type=int,
        default=1,
        help="Require at least this many available margin sources for BeeS-style scoring.",
    )
    parser.add_argument("--acc_target", type=float, default=0.5)
    parser.add_argument(
        "--public_threshold_quantile",
        type=float,
        default=0.25,
        help="Public samples must score no worse than this quantile of selected local samples.",
    )
    parser.add_argument(
        "--max_public_count",
        type=int,
        default=-1,
        help="Optional absolute cap for selected public samples. Negative disables the cap.",
    )
    parser.add_argument(
        "--max_public_ratio",
        type=float,
        default=-1.0,
        help="Optional target-size ratio cap for selected public samples. Negative disables the cap.",
    )
    parser.add_argument(
        "--max_local_ratio",
        type=float,
        default=-1.0,
        help="Optional target-size ratio cap for selected local samples. Enables mixed local/public ranking.",
    )
    parser.add_argument(
        "--min_margin",
        type=float,
        default=None,
        help="Drop samples with margin smaller than this threshold. Use to filter only extreme negative margins.",
    )
    parser.add_argument(
        "--drop_truncated",
        action="store_true",
        help="Drop samples whose client-margin scoring was truncated.",
    )
    parser.add_argument(
        "--auto_gate_hard",
        action="store_true",
        help="Keep hard samples only if their model score is competitive with local easy/medium candidates.",
    )
    parser.add_argument(
        "--hard_threshold_quantile",
        type=float,
        default=0.5,
        help="Hard samples must score at least this quantile of local non-hard candidates when --auto_gate_hard is used.",
    )
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--no_shuffle", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_stats = load_json(args.train_stats, default={}) or {}
    train_margin = safe_float(train_stats.get("margin", train_stats.get("margins")), 0.0)
    train_acc = safe_float(train_stats.get("acc"), args.acc_target)
    instability = max(0.0, -train_margin) + max(0.0, args.acc_target - train_acc)

    rows, bad_lines = prepare_rows(args.source, args, instability)
    rows, hard_score_threshold, hard_dropped = apply_hard_gate(rows, args)
    local_rows = [row for row in rows if row["_model_select_origin"] == "local"]
    public_rows = [row for row in rows if row["_model_select_origin"] == "public"]
    local_rows.sort(key=lambda x: (x["_model_select_score"], rng.random()), reverse=True)
    public_rows.sort(key=lambda x: (x["_model_select_score"], rng.random()), reverse=True)

    if args.max_local_ratio >= 0:
        local_cap = min(len(local_rows), int(args.target_size * args.max_local_ratio))
        ranked_rows = sorted(rows, key=lambda x: (x["_model_select_score"], rng.random()), reverse=True)
        selected = []
        selected_ids = set()
        local_count = 0

        for row in ranked_rows:
            if len(selected) >= args.target_size:
                break
            if row["_model_select_origin"] == "local" and local_count >= local_cap:
                continue
            selected.append(row)
            selected_ids.add(id(row))
            if row["_model_select_origin"] == "local":
                local_count += 1

        # If public candidates are insufficient, fill the remaining budget with
        # the best leftover rows so every client still receives target_size data.
        if len(selected) < args.target_size:
            for row in ranked_rows:
                if len(selected) >= args.target_size:
                    break
                if id(row) in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(id(row))

        selected_local = [row for row in selected if row["_model_select_origin"] == "local"]
        selected_public = [row for row in selected if row["_model_select_origin"] == "public"]
        threshold = None
    else:
        local_budget = min(args.target_size, len(local_rows))
        selected_local = local_rows[:local_budget]
        threshold = quantile(
            [row["_model_select_score"] for row in selected_local],
            args.public_threshold_quantile,
        )

        remaining = max(0, args.target_size - len(selected_local))
        if args.max_public_count >= 0:
            remaining = min(remaining, args.max_public_count)
        if args.max_public_ratio >= 0:
            remaining = min(remaining, int(args.target_size * args.max_public_ratio))
        if threshold is None:
            public_candidates = public_rows
        else:
            public_candidates = [row for row in public_rows if row["_model_select_score"] >= threshold]
        selected_public = public_candidates[:remaining]
        selected = selected_local + selected_public

    if args.dedup:
        selected = dedup_preserve_order(selected)
    if not args.no_shuffle:
        rng.shuffle(selected)

    margins = [row["_model_select_margin"] for row in selected]
    scores = [row["_model_select_score"] for row in selected]
    bees_probs = [
        row["_model_select_bees_prob"]
        for row in selected
        if row.get("_model_select_bees_prob") is not None
    ]
    bees_source_counts = [
        row["_model_select_bees_source_count"]
        for row in selected
        if row.get("_model_select_bees_source_count") is not None
    ]
    output_rows = [training_row(row) for row in selected]
    save_jsonl(output_rows, args.output)
    stats = {
        "output": args.output,
        "target_size": args.target_size,
        "num_output": len(selected),
        "bad_lines": bad_lines,
        "seed": args.seed,
        "train_stats_path": args.train_stats,
        "train_margin": train_margin,
        "train_acc": train_acc,
        "instability": instability,
        "tau": args.tau,
        "sigma": args.sigma,
        "risk_lambda": args.risk_lambda,
        "score_mode": args.score_mode,
        "bees_margin_keys": parse_csv_list(args.bees_margin_keys),
        "bees_margin_low": args.bees_margin_low,
        "bees_margin_high": args.bees_margin_high,
        "bees_min_prob": args.bees_min_prob,
        "bees_prob_floor": args.bees_prob_floor,
        "bees_min_sources": args.bees_min_sources,
        "public_threshold_quantile": args.public_threshold_quantile,
        "max_public_count": args.max_public_count,
        "max_public_ratio": args.max_public_ratio,
        "max_local_ratio": args.max_local_ratio,
        "min_margin": args.min_margin,
        "public_score_threshold": threshold,
        "auto_gate_hard": args.auto_gate_hard,
        "hard_threshold_quantile": args.hard_threshold_quantile,
        "hard_score_threshold": hard_score_threshold,
        "hard_dropped_by_gate": hard_dropped,
        "num_available_local": len(local_rows),
        "num_available_public": len(public_rows),
        "num_selected_local": len(selected_local),
        "num_selected_public": len(selected_public),
        "mean_selected_margin": mean(margins),
        "mean_selected_score": mean(scores),
        "mean_selected_bees_prob": mean(bees_probs),
        "mean_selected_bees_source_count": mean(bees_source_counts),
        "negative_margin_count": sum(1 for v in margins if v < 0),
        "source_counts": count_by(selected, "_model_select_source"),
        "origin_counts": count_by(selected, "_model_select_origin"),
        "difficulty_counts": count_by(selected, "_model_select_difficulty"),
    }
    stats_path = os.path.splitext(args.output)[0] + ".model_select_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("========== Model Select Curriculum Done ==========")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("==================================================")


if __name__ == "__main__":
    main()
