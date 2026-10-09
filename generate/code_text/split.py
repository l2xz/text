# encoding=utf-8
"""
Prepare easy/medium/hard curriculum pools from CodeXGLUE code-to-text DPO data.

Input:
  input_root/client_go/dpo_train.jsonl
  input_root/client_java/dpo_train.jsonl
  ...

Output:
  private_root/client_0/curri_easy.jsonl
  private_root/client_0/curri_medium.jsonl
  private_root/client_0/curri_hard_candidates.jsonl
  private_root/client_0/curri_hard_gap_selected.jsonl
  ...

Difficulty is derived from the selected model negative.  Higher similarity to
the reference docstring means a harder preference pair.  For heterogeneous
language clients, prefer the balanced-quantile strategy so each language is
bucketed by its own pair-quality distribution instead of fixed global BLEU
thresholds. This stage only forms candidate pools; model-aware selection later
decides which samples to actually train on.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_LANGUAGES = ["go", "java", "javascript", "php", "python", "ruby"]
LEVEL_TO_ID = {"easy": 1, "medium": 2, "hard": 3}
SPLIT_LEVELS = ("easy", "medium", "hard")


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def save_jsonl(rows: Iterable[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def metadata(row: Dict[str, object]) -> Dict[str, object]:
    meta = row.get("metadata", {})
    return meta if isinstance(meta, dict) else {}


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_]\w*", str(text).lower())


def code_like_ratio(text: str) -> float:
    text = str(text or "")
    if not text:
        return 0.0
    code_chars = sum(1 for ch in text if ch in "{}[]();=<>`/\\|")
    return code_chars / max(1, len(text))


def bad_docstring_reason(text: str) -> Optional[str]:
    text = str(text or "").strip()
    if not text:
        return "empty"
    lowered = text.lower()
    if "```" in text or "<code" in lowered or "</code" in lowered:
        return "code_block"
    if code_like_ratio(text) > 0.08:
        return "code_like"
    if re.search(r"\b(import|def|class|return|public|private|function|var|let|const)\s+[A-Za-z_]", text):
        return "contains_code"
    if re.search(r"\b(as an ai|i cannot|i'm sorry|cannot determine|not enough information)\b", lowered):
        return "assistant_refusal"
    if len(word_tokens(text)) < 3:
        return "too_short"
    return None


def is_safe_pair(row: Dict[str, object], args) -> bool:
    chosen = str(row.get("chosen_response", "")).strip()
    rejected = str(row.get("reject_response", "")).strip()
    if not chosen or not rejected or chosen == rejected:
        return False
    if bad_docstring_reason(chosen) is not None or bad_docstring_reason(rejected) is not None:
        return False

    meta = metadata(row)
    bleu = safe_float(meta.get("bleu_to_gold"), None)
    sim_char = safe_float(meta.get("sim_char"), None)
    sim_jacc = safe_float(meta.get("sim_jacc"), None)
    len_ratio = safe_float(meta.get("len_ratio"), None)

    if bleu is not None and not (args.min_bleu <= bleu <= args.max_bleu):
        return False
    if sim_char is not None and not (args.min_sim_char <= sim_char <= args.max_sim_char):
        return False
    if sim_jacc is not None and not (args.min_sim_jacc <= sim_jacc <= args.max_sim_jacc):
        return False
    if len_ratio is not None and len_ratio < args.min_len_ratio:
        return False
    return True


def filter_safe_pairs(rows: List[Dict[str, object]], args) -> List[Dict[str, object]]:
    if not args.safety_filter:
        return rows
    return [row for row in rows if is_safe_pair(row, args)]


def difficulty_score(row: Dict[str, object]) -> float:
    meta = metadata(row)
    bleu = safe_float(meta.get("bleu_to_gold"), None)
    if bleu is not None:
        return max(0.0, min(1.0, bleu / 100.0))

    sim_char = safe_float(meta.get("sim_char"), None)
    if sim_char is not None:
        return max(0.0, min(1.0, sim_char))

    sim_jacc = safe_float(meta.get("sim_jacc"), None)
    if sim_jacc is not None:
        return max(0.0, min(1.0, sim_jacc))

    hardness = safe_float(meta.get("hardness_score"), 0.0) or 0.0
    return hardness


def pair_quality_score(row: Dict[str, object]) -> float:
    """Prefer informative pairs: close enough to compare, not near-duplicates."""
    meta = metadata(row)
    bleu = clamp((safe_float(meta.get("bleu_to_gold"), 0.0) or 0.0) / 100.0, 0.0, 1.0)
    sim_char = clamp(safe_float(meta.get("sim_char"), 0.0) or 0.0, 0.0, 1.0)
    sim_jacc = clamp(safe_float(meta.get("sim_jacc"), 0.0) or 0.0, 0.0, 1.0)
    len_ratio = clamp(safe_float(meta.get("len_ratio"), 0.0) or 0.0, 0.0, 1.0)

    # Code-to-text negatives are most useful when they are semantically close
    # enough to be a real preference decision but not so close that the label
    # becomes noisy. These centers mirror the model-negative constructor.
    char_quality = 1.0 - min(1.0, abs(sim_char - 0.48) / 0.48)
    jacc_quality = 1.0 - min(1.0, abs(sim_jacc - 0.22) / 0.22)
    bleu_quality = 1.0 - min(1.0, abs(bleu - 0.24) / 0.24)
    return 0.35 * char_quality + 0.25 * jacc_quality + 0.25 * bleu_quality + 0.15 * len_ratio


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    q = clamp(q, 0.0, 1.0)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def annotate(row: Dict[str, object], level: str, origin: str, client_id: int, language: str) -> Dict[str, object]:
    out = dict(row)
    meta = dict(metadata(out))
    score = difficulty_score(out)
    meta.update(
        {
            "curriculum_difficulty": level,
            "curriculum_level": LEVEL_TO_ID[level],
            "curriculum_origin": origin,
            "curriculum_client_id": client_id,
            "curriculum_language": language,
            "curriculum_difficulty_score": round(score, 6),
        }
    )
    out["metadata"] = meta
    out["difficulty"] = level
    out["source"] = f"{origin}_{level}"
    out["origin"] = origin
    out["curriculum_level"] = LEVEL_TO_ID[level]
    return out


def split_by_ratio(
    rows: List[Dict[str, object]],
    easy_ratio: float,
    medium_ratio: float,
) -> Dict[str, List[Dict[str, object]]]:
    rows = sorted(rows, key=difficulty_score)
    n = len(rows)
    n_easy = int(round(n * easy_ratio))
    n_medium = int(round(n * medium_ratio))
    n_easy = max(0, min(n, n_easy))
    n_medium = max(0, min(n - n_easy, n_medium))
    return {
        "easy": rows[:n_easy],
        "medium": rows[n_easy : n_easy + n_medium],
        "hard": rows[n_easy + n_medium :],
    }


def split_by_balanced_quantile(
    rows: List[Dict[str, object]],
    easy_ratio: float,
    medium_ratio: float,
    quality_quantile: float,
    min_bucket_size: int,
) -> Dict[str, List[Dict[str, object]]]:
    if not rows:
        return {level: [] for level in SPLIT_LEVELS}

    scored = [(difficulty_score(row), pair_quality_score(row), row) for row in rows]
    quality_values = [item[1] for item in scored]
    quality_floor = quantile(quality_values, quality_quantile)

    filtered = [item for item in scored if item[1] >= quality_floor]
    if len(filtered) < max(3 * min_bucket_size, min(12, len(scored))):
        filtered = scored

    filtered = sorted(filtered, key=lambda item: (item[0], -item[1]))
    n = len(filtered)
    if n == 1:
        return {"easy": [filtered[0][2]], "medium": [], "hard": []}
    if n == 2:
        return {"easy": [filtered[0][2]], "medium": [filtered[1][2]], "hard": []}

    effective_min_bucket = min(max(1, min_bucket_size), max(1, n // 3))
    n_easy = int(round(n * easy_ratio))
    n_medium = int(round(n * medium_ratio))
    n_easy = max(effective_min_bucket, min(n - 2 * effective_min_bucket, n_easy))
    n_medium = max(effective_min_bucket, min(n - n_easy - effective_min_bucket, n_medium))

    easy = filtered[:n_easy]
    medium = filtered[n_easy : n_easy + n_medium]
    hard = filtered[n_easy + n_medium :]
    return {
        "easy": [item[2] for item in easy],
        "medium": [item[2] for item in medium],
        "hard": [item[2] for item in hard],
    }


def split_by_score_bands(
    rows: List[Dict[str, object]],
    easy_max_score: float,
    medium_min_score: float,
    medium_max_score: float,
    hard_min_score: float,
    hard_max_score: float,
) -> Dict[str, List[Dict[str, object]]]:
    splits = {"easy": [], "medium": [], "hard": []}
    ordered = sorted(rows, key=difficulty_score)
    for row in ordered:
        score = difficulty_score(row)
        if score <= easy_max_score:
            splits["easy"].append(row)
        elif medium_min_score <= score <= medium_max_score:
            splits["medium"].append(row)
        elif hard_min_score <= score <= hard_max_score:
            splits["hard"].append(row)

    # If a language is sparse after filtering, keep the pipeline runnable by
    # falling back to adjacent sorted buckets. The fallback only fills empty
    # pools; it does not override non-empty score bands.
    if ordered and not splits["easy"]:
        n = max(1, int(round(len(ordered) * 0.30)))
        splits["easy"] = ordered[:n]
    if ordered and not splits["medium"]:
        start = max(0, int(round(len(ordered) * 0.30)))
        end = max(start + 1, int(round(len(ordered) * 0.65)))
        splits["medium"] = ordered[start:end]
    if ordered and not splits["hard"]:
        start = max(0, int(round(len(ordered) * 0.65)))
        splits["hard"] = ordered[start:]

    return splits


def parse_language_dirs(input_root: str) -> List[Tuple[str, str]]:
    found = []
    for language in DEFAULT_LANGUAGES:
        path = os.path.join(input_root, f"client_{language}", "dpo_train.jsonl")
        if os.path.isfile(path):
            found.append((language, path))
    if found:
        return found

    for name in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, name, "dpo_train.jsonl")
        if name.startswith("client_") and os.path.isfile(path):
            found.append((name.replace("client_", ""), path))
    return found


def maybe_holdout_public(
    rows: List[Dict[str, object]],
    ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if ratio <= 0:
        return rows, []
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_public = int(round(len(shuffled) * ratio))
    n_public = max(0, min(len(shuffled), n_public))
    return shuffled[n_public:], shuffled[:n_public]


def write_client_pools(
    rows: List[Dict[str, object]],
    root: str,
    client_id: int,
    language: str,
    origin: str,
    easy_ratio: float,
    medium_ratio: float,
    args,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, Optional[float]]]]:
    client_dir = os.path.join(root, f"client_{client_id}")
    if args.split_strategy == "score_bands":
        splits = split_by_score_bands(
            rows,
            args.easy_max_score,
            args.medium_min_score,
            args.medium_max_score,
            args.hard_min_score,
            args.hard_max_score,
        )
    elif args.split_strategy == "balanced_quantile":
        splits = split_by_balanced_quantile(
            rows,
            easy_ratio,
            medium_ratio,
            args.balanced_quality_quantile,
            args.min_bucket_size,
        )
    else:
        splits = split_by_ratio(rows, easy_ratio, medium_ratio)
    annotated = {
        level: [annotate(row, level, origin, client_id, language) for row in level_rows]
        for level, level_rows in splits.items()
    }

    save_jsonl(annotated["easy"], os.path.join(client_dir, "curri_easy.jsonl"))
    save_jsonl(annotated["medium"], os.path.join(client_dir, "curri_medium.jsonl"))
    save_jsonl(annotated["hard"], os.path.join(client_dir, "curri_hard_candidates.jsonl"))
    save_jsonl(annotated["hard"], os.path.join(client_dir, "curri_hard_gap_selected.jsonl"))
    save_jsonl(
        annotated["easy"] + annotated["medium"] + annotated["hard"],
        os.path.join(client_dir, "curri_all_ordered.jsonl"),
    )
    counts = {level: len(items) for level, items in annotated.items()}
    split_stats = {
        level: {
            "count": len(items),
            "difficulty_min": min((difficulty_score(row) for row in items), default=None),
            "difficulty_max": max((difficulty_score(row) for row in items), default=None),
            "quality_mean": (
                sum(pair_quality_score(row) for row in items) / len(items) if items else None
            ),
        }
        for level, items in annotated.items()
    }
    return counts, split_stats


def write_empty_public(
    root: str, client_id: int
) -> Tuple[Dict[str, int], Dict[str, Dict[str, Optional[float]]]]:
    client_dir = os.path.join(root, f"client_{client_id}")
    save_jsonl([], os.path.join(client_dir, "curri_easy.jsonl"))
    save_jsonl([], os.path.join(client_dir, "curri_medium.jsonl"))
    save_jsonl([], os.path.join(client_dir, "curri_hard_candidates.jsonl"))
    save_jsonl([], os.path.join(client_dir, "curri_hard_gap_selected.jsonl"))
    save_jsonl([], os.path.join(client_dir, "curri_all_ordered.jsonl"))
    counts = {level: 0 for level in SPLIT_LEVELS}
    split_stats = {
        level: {"count": 0, "difficulty_min": None, "difficulty_max": None, "quality_mean": None}
        for level in SPLIT_LEVELS
    }
    return counts, split_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CodeXGLUE curriculum pools.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--private_root", required=True)
    parser.add_argument("--public_root", default=None)
    parser.add_argument("--public_mode", choices=["empty", "holdout"], default="empty")
    parser.add_argument("--public_ratio", type=float, default=0.0)
    parser.add_argument("--easy_ratio", type=float, default=0.4)
    parser.add_argument("--medium_ratio", type=float, default=0.4)
    parser.add_argument(
        "--split_strategy",
        choices=["score_bands", "ratio", "balanced_quantile"],
        default="score_bands",
    )
    parser.add_argument("--easy_max_score", type=float, default=0.16)
    parser.add_argument("--medium_min_score", type=float, default=0.16)
    parser.add_argument("--medium_max_score", type=float, default=0.30)
    parser.add_argument("--hard_min_score", type=float, default=0.30)
    parser.add_argument("--hard_max_score", type=float, default=0.45)
    parser.add_argument("--balanced_quality_quantile", type=float, default=0.05)
    parser.add_argument("--balanced_lower_quantile", type=float, default=None)
    parser.add_argument("--balanced_upper_quantile", type=float, default=None)
    parser.add_argument("--min_bucket_size", type=int, default=1)
    parser.add_argument("--max_rows_per_client", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety_filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_bleu", type=float, default=0.0)
    parser.add_argument("--max_bleu", type=float, default=45.0)
    parser.add_argument("--min_sim_char", type=float, default=0.10)
    parser.add_argument("--max_sim_char", type=float, default=0.82)
    parser.add_argument("--min_sim_jacc", type=float, default=0.04)
    parser.add_argument("--max_sim_jacc", type=float, default=0.52)
    parser.add_argument("--min_len_ratio", type=float, default=0.45)
    args = parser.parse_args()

    if args.easy_ratio < 0 or args.medium_ratio < 0 or args.easy_ratio + args.medium_ratio > 1:
        raise ValueError("Require easy_ratio >= 0, medium_ratio >= 0, and easy_ratio + medium_ratio <= 1")
    if args.public_mode == "holdout" and not args.public_root:
        raise ValueError("--public_root is required when --public_mode=holdout")
    if args.balanced_lower_quantile is not None:
        args.balanced_quality_quantile = args.balanced_lower_quantile

    rng = random.Random(args.seed)
    language_dirs = parse_language_dirs(args.input_root)
    if not language_dirs:
        raise FileNotFoundError(f"No client_*/dpo_train.jsonl found under: {args.input_root}")

    stats = {
        "input_root": args.input_root,
        "private_root": args.private_root,
        "public_root": args.public_root,
        "public_mode": args.public_mode,
        "public_ratio": args.public_ratio,
        "split_strategy": args.split_strategy,
        "easy_ratio": args.easy_ratio,
        "medium_ratio": args.medium_ratio,
        "hard_ratio": 1.0 - args.easy_ratio - args.medium_ratio,
        "score_bands": {
            "easy": [0.0, args.easy_max_score],
            "medium": [args.medium_min_score, args.medium_max_score],
            "hard": [args.hard_min_score, args.hard_max_score],
        },
        "balanced_quantile": {
            "quality_quantile": args.balanced_quality_quantile,
            "min_bucket_size": args.min_bucket_size,
        },
        "clients": {},
        "client_id_mapping": {},
    }

    for client_id, (language, path) in enumerate(language_dirs):
        rows = read_jsonl(path)
        if args.max_rows_per_client > 0:
            rows = rows[: args.max_rows_per_client]
        num_raw_rows = len(rows)
        rows = filter_safe_pairs(rows, args)

        private_rows, public_rows = rows, []
        if args.public_mode == "holdout":
            private_rows, public_rows = maybe_holdout_public(rows, args.public_ratio, rng)

        private_counts, private_split_stats = write_client_pools(
            private_rows,
            args.private_root,
            client_id,
            language,
            "local",
            args.easy_ratio,
            args.medium_ratio,
            args,
        )

        public_counts = None
        public_split_stats = None
        if args.public_root:
            if args.public_mode == "holdout":
                public_counts, public_split_stats = write_client_pools(
                    public_rows,
                    args.public_root,
                    client_id,
                    language,
                    "public",
                    args.easy_ratio,
                    args.medium_ratio,
                    args,
                )
            else:
                public_counts, public_split_stats = write_empty_public(args.public_root, client_id)

        stats["clients"][f"client_{client_id}"] = {
            "language": language,
            "source_file": path,
            "num_input": num_raw_rows,
            "num_after_safety_filter": len(rows),
            "safety_filter": args.safety_filter,
            "private_counts": private_counts,
            "private_split_stats": private_split_stats,
            "public_counts": public_counts,
            "public_split_stats": public_split_stats,
        }
        stats["client_id_mapping"][f"client_{client_id}"] = language

    save_json(stats, os.path.join(args.private_root, "curriculum_pool_stats.json"))
    save_json(stats["client_id_mapping"], os.path.join(args.private_root, "client_id_mapping.json"))
    if args.public_root:
        save_json(stats, os.path.join(args.public_root, "curriculum_pool_stats.json"))
        save_json(stats["client_id_mapping"], os.path.join(args.public_root, "client_id_mapping.json"))

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
