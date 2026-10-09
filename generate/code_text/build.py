# encoding=utf-8
"""
Construct DPO data from google/code_x_glue_ct_code_to_text.

The dataset is naturally federated by programming language:
go, java, javascript, php, python, and ruby.  For each sample, the chosen
response is the gold docstring.  The rejected response is either a model
candidate from a generated-candidates jsonl file or a hard retrieval negative:
a docstring from a different function whose code tokens are similar to the
current code.

Output format matches the repository DPO trainers:
{
  "query": "...",
  "chosen_response": "...",
  "reject_response": "...",
  ...
}
"""

import argparse
import collections
import json
import math
import os
import random
import re
from difflib import SequenceMatcher

from tqdm import tqdm


DEFAULT_DATASET = "google/code_x_glue_ct_code_to_text"
DEFAULT_LANGUAGES = ["go", "java", "javascript", "php", "python", "ruby"]
SPECIAL_DOC_PATTERNS = [r"<img\b", r"<a\b", r"</?\w+[^>]*>", r"https?://", r"www\."]


def read_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text):
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"```(?:\w+)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_docstring(text):
    text = normalize_text(text)
    text = re.sub(r"^\s*(//+|#+)\s*", "", text)
    text = re.sub(r"^\s*/\*+\s*", "", text)
    text = re.sub(r"\s*\*/\s*$", "", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\*\s?", "", line).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
    return normalize_text(text)


def token_len(text):
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def word_tokens(text):
    return re.findall(r"[A-Za-z_]\w*", normalize_text(text).lower())


def code_like_ratio(text):
    text = normalize_text(text)
    if not text:
        return 0.0
    code_chars = sum(1 for ch in text if ch in "{}[]();=<>`/\\|")
    return code_chars / max(1, len(text))


def bad_docstring_reason(text):
    text = clean_docstring(text)
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


def field_tokens(row, key, fallback_text=""):
    value = row.get(key)
    if isinstance(value, list):
        return [normalize_text(x).lower() for x in value if normalize_text(x)]
    return word_tokens(value if value is not None else fallback_text)


def is_probably_english(text):
    text = normalize_text(text)
    letters = re.findall(r"[A-Za-z]", text)
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return bool(letters) and non_ascii / max(1, len(text)) < 0.08


def has_special_doc_tokens(text):
    text = normalize_text(text).lower()
    return any(re.search(pattern, text) for pattern in SPECIAL_DOC_PATTERNS)


def passes_codexglue_doc_filter(row, min_doc_tokens, max_doc_tokens):
    doc = clean_docstring(row.get("docstring", ""))
    doc_tokens = field_tokens(row, "docstring_tokens", doc)
    if len(doc_tokens) < min_doc_tokens or len(doc_tokens) > max_doc_tokens:
        return False
    if has_special_doc_tokens(doc):
        return False
    if not is_probably_english(doc):
        return False
    return True


def dedupe_keep_order(texts):
    out = []
    seen = set()
    for text in texts:
        text = clean_docstring(text)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def char_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity(a, b):
    a_tokens = set(re.findall(r"\w+", a.lower(), flags=re.UNICODE))
    b_tokens = set(re.findall(r"\w+", b.lower(), flags=re.UNICODE))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def token_jaccard(a_tokens, b_tokens):
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, len(a_set | b_set))


def ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def smoothed_bleu(reference, prediction, max_order=4):
    ref = word_tokens(reference)
    pred = word_tokens(prediction)
    if not ref or not pred:
        return 0.0
    precisions = []
    for n in range(1, max_order + 1):
        ref_counts = collections.Counter(ngrams(ref, n))
        pred_counts = collections.Counter(ngrams(pred, n))
        overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        total = max(1, sum(pred_counts.values()))
        precisions.append((overlap + 1.0) / (total + 1.0))
    log_precision = sum(math.log(p) for p in precisions) / max_order
    brevity = 1.0 if len(pred) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(pred)))
    return 100.0 * brevity * math.exp(log_precision)


def build_query(row, language):
    code = normalize_text(row.get("code", row.get("original_string", "")))
    func_name = normalize_text(row.get("func_name", ""))
    header = [
        "Generate a concise English docstring that summarizes the following code.",
        "Return only the docstring text.",
        f"Language: {language}",
    ]
    if func_name:
        header.append(f"Function: {func_name}")
    header.extend(["", "Code:", f"```{language}", code, "```", "", "Docstring:"])
    return "\n".join(header)


def make_sample_id(language, split, row, fallback_idx):
    raw_id = row.get("id", fallback_idx)
    return f"{language}:{split}:{raw_id}"


def load_language_split(dataset_name, language, split, cache_dir=None, local_dataset_root=None):
    from datasets import load_dataset, load_from_disk

    if local_dataset_root:
        path = os.path.join(local_dataset_root, language)
        try:
            ds = load_from_disk(path)
            return ds[split] if hasattr(ds, "keys") and split in ds else ds
        except Exception:
            pass
    return load_dataset(dataset_name, language, split=split, cache_dir=cache_dir)


def build_generated_map(rows):
    out = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            out[str(sample_id)] = row
    return out


def hardness_score(
    chosen,
    rejected,
    min_bleu=0.0,
    max_bleu=70.0,
    target_bleu=28.0,
    min_sim_char=0.05,
    max_sim_char=0.90,
    min_sim_jacc=0.02,
    max_sim_jacc=0.70,
    min_len_ratio=0.25,
):
    chosen = clean_docstring(chosen)
    rejected = clean_docstring(rejected)
    if not chosen or not rejected or chosen == rejected:
        return None
    if bad_docstring_reason(chosen) is not None:
        return None
    if bad_docstring_reason(rejected) is not None:
        return None
    chosen_len = token_len(chosen)
    rejected_len = token_len(rejected)
    if chosen_len == 0 or rejected_len == 0:
        return None
    len_ratio = min(chosen_len, rejected_len) / max(chosen_len, rejected_len)
    if len_ratio < min_len_ratio:
        return None
    sim_char = char_similarity(chosen, rejected)
    sim_jacc = jaccard_similarity(chosen, rejected)
    bleu = smoothed_bleu(chosen, rejected)
    if sim_char > 0.985:
        return None
    if sim_char < 0.05 and sim_jacc < 0.02:
        return None
    if not (min_bleu <= bleu <= max_bleu):
        return None
    if not (min_sim_char <= sim_char <= max_sim_char):
        return None
    if not (min_sim_jacc <= sim_jacc <= max_sim_jacc):
        return None

    # Prefer negatives that are plausible and safely different from the
    # reference. Very high BLEU/similarity candidates are often correct
    # paraphrases, so the target band is intentionally moderate.
    bleu_span = max(target_bleu - min_bleu, max_bleu - target_bleu, 1.0)
    bleu_band_score = 1.0 - min(1.0, abs(bleu - target_bleu) / bleu_span)
    score = 0.0
    score += 1.20 * bleu_band_score
    score += 0.80 - abs(sim_char - 0.48)
    score += 0.65 - abs(sim_jacc - 0.22)
    score += 0.35 * len_ratio
    return {
        "candidate": rejected,
        "score": score,
        "sim_char": sim_char,
        "sim_jacc": sim_jacc,
        "bleu": bleu,
        "len_ratio": len_ratio,
    }


def pick_model_negative(chosen, generated_row, args):
    if not generated_row:
        return None
    candidates = dedupe_keep_order(generated_row.get("generated_responses", []))
    scored = []
    for cand in candidates:
        item = hardness_score(
            chosen,
            cand,
            min_bleu=args.model_negative_min_bleu,
            max_bleu=args.model_negative_max_bleu,
            target_bleu=args.model_negative_target_bleu,
            min_sim_char=args.model_negative_min_sim_char,
            max_sim_char=args.model_negative_max_sim_char,
            min_sim_jacc=args.model_negative_min_sim_jacc,
            max_sim_jacc=args.model_negative_max_sim_jacc,
            min_len_ratio=args.model_negative_min_len_ratio,
        )
        if item is not None:
            scored.append(item)
    if not scored:
        return None
    return max(scored, key=lambda x: x["score"])


def row_identity(row, fallback_idx):
    return (
        str(row.get("repo", "")),
        str(row.get("path", "")),
        str(row.get("func_name", "")),
        str(row.get("id", fallback_idx)),
    )


def build_retrieval_pool(rows, min_doc_tokens, max_doc_tokens):
    pool = []
    for idx, row in enumerate(rows):
        if not passes_codexglue_doc_filter(row, min_doc_tokens, max_doc_tokens):
            continue
        code = normalize_text(row.get("code", row.get("original_string", "")))
        pool.append(
            {
                "identity": row_identity(row, idx),
                "docstring": clean_docstring(row.get("docstring", "")),
                "code_tokens": field_tokens(row, "code_tokens", code),
                "func_name": normalize_text(row.get("func_name", "")).lower(),
                "repo": row.get("repo", ""),
            }
        )
    return pool


def pick_mismatched_negative(chosen, row, row_idx, pool, rng, pool_size):
    if not pool:
        return None
    current_code = normalize_text(row.get("code", row.get("original_string", "")))
    current_code_tokens = field_tokens(row, "code_tokens", current_code)
    current_func_name = normalize_text(row.get("func_name", "")).lower()
    current_identity = row_identity(row, row_idx)
    scored_pool = []
    for item in pool:
        if item["identity"] == current_identity:
            continue
        code_sim = token_jaccard(current_code_tokens, item["code_tokens"])
        name_bonus = 0.15 if current_func_name and current_func_name == item["func_name"] else 0.0
        repo_penalty = 0.05 if row.get("repo", "") == item.get("repo", "") else 0.0
        scored_pool.append((code_sim + name_bonus - repo_penalty, item))
    if not scored_pool:
        return None
    scored_pool.sort(key=lambda x: x[0], reverse=True)
    top = [item for _, item in scored_pool[: max(pool_size * 4, pool_size)]]
    candidates = rng.sample(top, min(pool_size, len(top)))
    scored = []
    for cand in candidates:
        item = hardness_score(chosen, cand["docstring"], max_bleu=55.0, target_bleu=24.0, min_len_ratio=0.35)
        if item is not None:
            item["code_similarity"] = token_jaccard(current_code_tokens, cand["code_tokens"])
            scored.append(item)
    if not scored:
        return None
    return max(scored, key=lambda x: (x.get("code_similarity", 0.0), x["score"]))


def synthetic_negative(chosen):
    chosen = clean_docstring(chosen)
    candidates = []
    if re.search(r"\breturns?\b", chosen, flags=re.IGNORECASE):
        candidates.append(re.sub(r"\breturns?\b", "validates", chosen, count=1, flags=re.IGNORECASE))
    if re.search(r"\bcreates?\b", chosen, flags=re.IGNORECASE):
        candidates.append(re.sub(r"\bcreates?\b", "deletes", chosen, count=1, flags=re.IGNORECASE))
    words = chosen.split()
    if len(words) >= 6:
        swapped = words[:]
        swapped[-1], swapped[-2] = swapped[-2], swapped[-1]
        candidates.append(" ".join(swapped))
    for cand in candidates:
        item = hardness_score(chosen, cand)
        if item is not None:
            return item
    return None


def construct_language_dpo(
    dataset,
    language,
    split,
    generated_map,
    rng,
    max_samples,
    min_doc_tokens,
    max_doc_tokens,
    negative_source,
    hard_negative_pool,
    apply_codexglue_filters,
    args,
):
    rows = list(dataset)
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]

    if apply_codexglue_filters:
        rows = [row for row in rows if passes_codexglue_doc_filter(row, min_doc_tokens, max_doc_tokens)]

    retrieval_pool = build_retrieval_pool(rows, min_doc_tokens, max_doc_tokens)

    output = []
    for idx, row in enumerate(tqdm(rows, desc=f"{language}:{split}")):
        chosen = clean_docstring(row.get("docstring", ""))
        chosen_tokens = len(field_tokens(row, "docstring_tokens", chosen))
        if chosen_tokens < min_doc_tokens or chosen_tokens > max_doc_tokens:
            continue

        sample_id = make_sample_id(language, split, row, idx)
        reject = None
        reject_source = None

        if negative_source in {"model", "mixed"}:
            reject = pick_model_negative(chosen, generated_map.get(sample_id), args)
            if reject is not None:
                reject_source = "model_sample"

        if reject is None and negative_source in {"mismatch", "mixed"}:
            reject = pick_mismatched_negative(chosen, row, idx, retrieval_pool, rng, hard_negative_pool)
            if reject is not None:
                reject_source = "code_retrieval_docstring"

        if reject is None and negative_source in {"synthetic", "mixed"}:
            reject = synthetic_negative(chosen)
            if reject is not None:
                reject_source = "synthetic"

        if reject is None:
            continue

        output.append(
            {
                "sample_id": sample_id,
                "query": build_query(row, language),
                "chosen_response": chosen,
                "reject_response": reject["candidate"],
                "chosen_length": chosen_tokens,
                "reject_length": token_len(reject["candidate"]),
                "metadata": {
                    "dataset": DEFAULT_DATASET,
                    "language": language,
                    "split": split,
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "func_name": row.get("func_name", ""),
                    "url": row.get("url", ""),
                    "reject_source": reject_source,
                    "sim_char": round(reject["sim_char"], 4),
                    "sim_jacc": round(reject["sim_jacc"], 4),
                    "bleu_to_gold": round(reject.get("bleu", 0.0), 4),
                    "code_similarity": round(reject.get("code_similarity", 0.0), 4),
                    "len_ratio": round(reject["len_ratio"], 4),
                    "hardness_score": round(reject["score"], 4),
                },
            }
        )

    return output


def parse_languages(text):
    if not text or text == "all":
        return DEFAULT_LANGUAGES
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--local_dataset_root", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--languages", type=str, default="all", help="Comma/space list or 'all'.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--generated_candidates", type=str, default=None)
    parser.add_argument(
        "--negative_source",
        choices=["mismatch", "model", "synthetic", "mixed"],
        default="mismatch",
        help="Use mismatched gold docstrings, model generations, synthetic corruptions, or mixed fallback.",
    )
    parser.add_argument("--max_samples_per_language", type=int, default=-1)
    parser.add_argument("--min_doc_tokens", type=int, default=3)
    parser.add_argument("--max_doc_tokens", type=int, default=256)
    parser.add_argument("--hard_negative_pool", type=int, default=64)
    parser.add_argument(
        "--model_negative_min_bleu",
        type=float,
        default=0.0,
        help="Minimum BLEU-to-reference for a model sample to be used as rejected.",
    )
    parser.add_argument(
        "--model_negative_max_bleu",
        type=float,
        default=45.0,
        help="Maximum BLEU-to-reference for a model sample to be used as rejected. Lower values avoid rejecting likely-correct paraphrases.",
    )
    parser.add_argument(
        "--model_negative_target_bleu",
        type=float,
        default=25.0,
        help="Preferred BLEU-to-reference band center for model rejected samples.",
    )
    parser.add_argument("--model_negative_min_sim_char", type=float, default=0.10)
    parser.add_argument("--model_negative_max_sim_char", type=float, default=0.82)
    parser.add_argument("--model_negative_min_sim_jacc", type=float, default=0.04)
    parser.add_argument("--model_negative_max_sim_jacc", type=float, default=0.52)
    parser.add_argument("--model_negative_min_len_ratio", type=float, default=0.45)
    parser.add_argument(
        "--no_codexglue_filters",
        action="store_true",
        help="Disable CodeXGLUE-style docstring filters.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--merge_output",
        action="store_true",
        help="Also write all languages to output_root/dpo_train.jsonl.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    languages = parse_languages(args.languages)
    generated_rows = read_jsonl(args.generated_candidates)
    generated_map = build_generated_map(generated_rows)

    all_rows = []
    stats = {}
    os.makedirs(args.output_root, exist_ok=True)

    for language in languages:
        dataset = load_language_split(
            args.dataset_name,
            language,
            args.split,
            cache_dir=args.cache_dir,
            local_dataset_root=args.local_dataset_root,
        )
        dpo_rows = construct_language_dpo(
            dataset=dataset,
            language=language,
            split=args.split,
            generated_map=generated_map,
            rng=rng,
            max_samples=args.max_samples_per_language,
            min_doc_tokens=args.min_doc_tokens,
            max_doc_tokens=args.max_doc_tokens,
            negative_source=args.negative_source,
            hard_negative_pool=args.hard_negative_pool,
            apply_codexglue_filters=not args.no_codexglue_filters,
            args=args,
        )

        client_dir = os.path.join(args.output_root, f"client_{language}")
        save_jsonl(dpo_rows, os.path.join(client_dir, "dpo_train.jsonl"))
        stats[language] = {
            "num_dpo_pairs": len(dpo_rows),
            "output": os.path.join(client_dir, "dpo_train.jsonl"),
        }
        all_rows.extend(dpo_rows)

    if args.merge_output:
        save_jsonl(all_rows, os.path.join(args.output_root, "dpo_train.jsonl"))

    with open(os.path.join(args.output_root, "construct_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset_name,
                "split": args.split,
                "languages": languages,
                "negative_source": args.negative_source,
                "model_negative_filter": {
                    "min_bleu": args.model_negative_min_bleu,
                    "max_bleu": args.model_negative_max_bleu,
                    "target_bleu": args.model_negative_target_bleu,
                    "min_sim_char": args.model_negative_min_sim_char,
                    "max_sim_char": args.model_negative_max_sim_char,
                    "min_sim_jacc": args.model_negative_min_sim_jacc,
                    "max_sim_jacc": args.model_negative_max_sim_jacc,
                    "min_len_ratio": args.model_negative_min_len_ratio,
                },
                "codexglue_filters": not args.no_codexglue_filters,
                "num_total_pairs": len(all_rows),
                "by_language": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved CodeXGLUE DPO data to: {args.output_root}")


if __name__ == "__main__":
    main()
