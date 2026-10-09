# encoding=utf-8
"""
Generate model candidates for CodeXGLUE code-to-text DPO construction.

Each output row keeps the gold docstring as reference_output and stores several
model-generated docstrings.  The paired DPO constructor can then use:
  chosen = reference_output
  rejected = a similar but non-equivalent model sample
"""

import argparse
import json
import math
import os
import re

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams


DEFAULT_DATASET = "google/code_x_glue_ct_code_to_text"
DEFAULT_LANGUAGES = ["go", "java", "javascript", "php", "python", "ruby"]
SPECIAL_DOC_PATTERNS = [r"<img\b", r"<a\b", r"</?\w+[^>]*>", r"https?://", r"www\."]
SYSTEM_PROMPT = "You are a careful code documentation assistant."


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
    return normalize_text("\n".join(lines))


def word_tokens(text):
    return re.findall(r"[A-Za-z_]\w*", normalize_text(text).lower())


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


def parse_languages(text):
    if not text or text == "all":
        return DEFAULT_LANGUAGES
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def load_language_split(dataset_name, language, split, cache_dir=None, local_dataset_root=None):
    if local_dataset_root:
        path = os.path.join(local_dataset_root, language)
        ds = load_from_disk(path)
        return ds[split] if hasattr(ds, "keys") and split in ds else ds
    return load_dataset(dataset_name, language, split=split, cache_dir=cache_dir)


def make_sample_id(language, split, row, fallback_idx):
    raw_id = row.get("id", fallback_idx)
    return f"{language}:{split}:{raw_id}"


def build_user_prompt(row, language, code_override=None):
    code = normalize_text(code_override if code_override is not None else row.get("code", row.get("original_string", "")))
    func_name = normalize_text(row.get("func_name", ""))
    parts = [
        "Generate a concise English docstring that summarizes the following code.",
        "Return only the docstring text.",
        f"Language: {language}",
    ]
    if func_name:
        parts.append(f"Function: {func_name}")
    parts.extend(["", "Code:", f"```{language}", code, "```", "", "Docstring:"])
    return "\n".join(parts)


def build_qwen_prompt(user_msg):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_code_by_tokens(tokenizer, code, max_code_tokens):
    code = normalize_text(code)
    token_ids = tokenizer.encode(code, add_special_tokens=False)
    if len(token_ids) <= max_code_tokens:
        return code, False, len(token_ids), len(token_ids)

    if max_code_tokens <= 16:
        kept = token_ids[:max_code_tokens]
        return tokenizer.decode(kept, skip_special_tokens=True), True, len(token_ids), len(kept)

    head_tokens = max(8, int(max_code_tokens * 0.65))
    tail_tokens = max(8, max_code_tokens - head_tokens)
    if head_tokens + tail_tokens > max_code_tokens:
        tail_tokens = max(1, max_code_tokens - head_tokens)

    kept = token_ids[:head_tokens] + token_ids[-tail_tokens:]
    text = tokenizer.decode(token_ids[:head_tokens], skip_special_tokens=True)
    text += "\n\n# ... truncated long code ...\n\n"
    text += tokenizer.decode(token_ids[-tail_tokens:], skip_special_tokens=True)
    return text, True, len(token_ids), len(kept)


def build_prompt_with_budget(row, language, tokenizer, max_prompt_tokens):
    raw_user_msg = build_user_prompt(row, language)
    raw_prompt = build_qwen_prompt(raw_user_msg)
    raw_prompt_tokens = count_tokens(tokenizer, raw_prompt)
    if max_prompt_tokens <= 0 or raw_prompt_tokens <= max_prompt_tokens:
        return raw_user_msg, raw_prompt, {
            "prompt_truncated": False,
            "prompt_tokens_raw": raw_prompt_tokens,
            "prompt_tokens_final": raw_prompt_tokens,
            "code_tokens_raw": count_tokens(tokenizer, normalize_text(row.get("code", row.get("original_string", "")))),
            "code_tokens_final": count_tokens(tokenizer, normalize_text(row.get("code", row.get("original_string", "")))),
        }

    empty_prompt = build_qwen_prompt(build_user_prompt(row, language, code_override=""))
    overhead_tokens = count_tokens(tokenizer, empty_prompt)
    code_budget = max(32, max_prompt_tokens - overhead_tokens - 16)
    code = normalize_text(row.get("code", row.get("original_string", "")))

    for _ in range(8):
        truncated_code, was_truncated, code_tokens_raw, code_tokens_final = truncate_code_by_tokens(
            tokenizer, code, code_budget
        )
        user_msg = build_user_prompt(row, language, code_override=truncated_code)
        prompt = build_qwen_prompt(user_msg)
        final_tokens = count_tokens(tokenizer, prompt)
        if final_tokens <= max_prompt_tokens or code_budget <= 32:
            return user_msg, prompt, {
                "prompt_truncated": was_truncated,
                "prompt_tokens_raw": raw_prompt_tokens,
                "prompt_tokens_final": final_tokens,
                "code_tokens_raw": code_tokens_raw,
                "code_tokens_final": code_tokens_final,
            }
        code_budget = max(32, int(code_budget * 0.85))

    return user_msg, prompt, {
        "prompt_truncated": True,
        "prompt_tokens_raw": raw_prompt_tokens,
        "prompt_tokens_final": final_tokens,
        "code_tokens_raw": code_tokens_raw,
        "code_tokens_final": code_tokens_final,
    }


def select_shard(rows, num_shards, shard_id):
    if num_shards <= 1:
        return rows
    shard_size = math.ceil(len(rows) / num_shards)
    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, len(rows))
    return rows[start_idx:end_idx]


def collect_rows(args):
    rows = []
    for language in parse_languages(args.languages):
        dataset = load_language_split(
            args.dataset_name,
            language,
            args.split,
            cache_dir=args.cache_dir,
            local_dataset_root=args.local_dataset_root,
        )
        language_rows = []
        for idx, row in enumerate(tqdm(dataset, desc=f"load:{language}:{args.split}")):
            row = dict(row)
            if not args.no_codexglue_filters and not passes_codexglue_doc_filter(
                row, args.min_doc_tokens, args.max_doc_tokens
            ):
                continue
            if not clean_docstring(row.get("docstring", "")):
                continue
            language_rows.append((language, idx, row))
            if args.max_samples_per_language > 0 and len(language_rows) >= args.max_samples_per_language:
                break
        rows.extend(language_rows)
    return select_shard(rows, args.num_shards, args.shard_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--local_dataset_root", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--languages", type=str, default="all")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=0,
        help="Truncate prompts to this token budget before generation. Default uses max_model_len - max_tokens - 32.",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_samples_per_language", type=int, default=-1)
    parser.add_argument("--min_doc_tokens", type=int, default=3)
    parser.add_argument("--max_doc_tokens", type=int, default=256)
    parser.add_argument("--no_codexglue_filters", action="store_true")
    args = parser.parse_args()

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id must be in [0, {args.num_shards}), got {args.shard_id}")

    rows = collect_rows(args)
    print(f"[Generate] shard={args.shard_id}/{args.num_shards}, examples={len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    max_prompt_tokens = args.max_prompt_tokens
    if max_prompt_tokens <= 0:
        max_prompt_tokens = max(128, args.max_model_len - args.max_tokens - 32)
    print(f"[Generate] max_prompt_tokens={max_prompt_tokens}")

    prompts = []
    metadata = []
    num_truncated = 0
    for language, idx, row in rows:
        user_msg, prompt, prompt_info = build_prompt_with_budget(row, language, tokenizer, max_prompt_tokens)
        prompts.append(prompt)
        if prompt_info["prompt_truncated"]:
            num_truncated += 1
        metadata.append(
            {
                "sample_id": make_sample_id(language, args.split, row, idx),
                "language": language,
                "split": args.split,
                "instruction": "Generate a concise English docstring that summarizes the given code.",
                "input": normalize_text(row.get("code", row.get("original_string", ""))),
                "reference_output": clean_docstring(row.get("docstring", "")),
                "prompt_messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "metadata": {
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "func_name": row.get("func_name", ""),
                    "url": row.get("url", ""),
                    "prompt_truncated": prompt_info["prompt_truncated"],
                    "prompt_tokens_raw": prompt_info["prompt_tokens_raw"],
                    "prompt_tokens_final": prompt_info["prompt_tokens_final"],
                    "code_tokens_raw": prompt_info["code_tokens_raw"],
                    "code_tokens_final": prompt_info["code_tokens_final"],
                },
            }
        )
    print(f"[Generate] truncated_prompts={num_truncated}/{len(prompts)}")

    print("[Generate] Initializing vLLM ...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>", "</s>"],
    )

    print("[Generate] Sampling candidate docstrings ...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    os.makedirs(args.output_root, exist_ok=True)
    output_file = os.path.join(args.output_root, f"generated_candidates_part{args.shard_id}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for meta, output in zip(metadata, outputs):
            generated_responses = []
            for item in output.outputs:
                text = clean_docstring(item.text)
                if text:
                    generated_responses.append(text)
            row = dict(meta)
            row["generated_responses"] = generated_responses
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Generate] Saved candidates to: {output_file}")


if __name__ == "__main__":
    main()
