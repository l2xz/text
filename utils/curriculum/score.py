# encoding=utf-8
import argparse
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(f"[Warning] Bad JSON skipped: {path}, line={line_id}, error={exc}")
    return rows


def limit_rows(rows: List[Dict[str, Any]], max_rows: int, mode: str, seed: int, source_name: str) -> List[Dict[str, Any]]:
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    if mode == "random":
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(rows)), max_rows))
        limited = [rows[i] for i in indices]
    else:
        limited = rows[:max_rows]
    print(f"[Limit] {source_name}: keep {len(limited)}/{len(rows)} rows, mode={mode}, seed={seed}")
    return limited


def save_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_named_path(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid named path '{spec}', expected name=/path/file.jsonl")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid named path '{spec}', empty name")
    return name, path


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for msg in value:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"]).strip()
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def get_response(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def format_prompt(tokenizer, system: str, query: str, use_chat_template: bool) -> str:
    if use_chat_template:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": query})
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    if system:
        return f"<|system|>\n{system}\n<|user|>\n{query}\n<|assistant|>\n"
    return f"<|user|>\n{query}\n<|assistant|>\n"


def token_logprob(entry: Any, token_id: int) -> Optional[float]:
    if entry is None:
        return None
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, dict):
        for key in (token_id, str(token_id)):
            value = entry.get(key)
            if value is None:
                continue
            if hasattr(value, "logprob"):
                return float(value.logprob)
            try:
                return float(value)
            except Exception:
                pass
        if len(entry) == 1:
            value = next(iter(entry.values()))
            if hasattr(value, "logprob"):
                return float(value.logprob)
            try:
                return float(value)
            except Exception:
                return None
    return None


def build_examples(rows: List[Dict[str, Any]], tokenizer, args, source_name: str):
    examples = []
    scored = []
    skipped = 0
    for row in rows:
        query = extract_text(row.get(args.prompt_key, row.get("prompt", "")))
        system = extract_text(row.get(args.system_key, "")) if args.system_key else ""
        chosen = get_response(row, [args.chosen_key, "chosen", "chosen_response", "chosen_text"])
        rejected = get_response(row, [args.rejected_key, "rejected", "reject_response", "rejected_response", "reject_text"])

        out = dict(row)
        out["_client_margin_source"] = source_name
        if not query or not chosen or not rejected:
            out["client_margin_valid"] = False
            out["client_margin_error"] = "missing query/chosen/rejected"
            scored.append(out)
            skipped += 1
            continue

        prompt_text = format_prompt(tokenizer, system, query, args.use_chat_template)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        for response_name, response_text in (("chosen", chosen), ("rejected", rejected)):
            full_text = prompt_text + response_text
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            truncated = len(full_ids) > args.max_model_len
            prompt_token_ids = full_ids[: args.max_model_len]
            examples.append(
                {
                    "row_index": len(scored),
                    "response_name": response_name,
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_len": min(len(prompt_ids), args.max_model_len),
                    "full_len": len(prompt_token_ids),
                    "truncated": truncated,
                }
            )
        scored.append(out)
    return examples, scored, skipped


def batched(items: List[Dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def sum_response_logprobs(
    llm: LLM,
    examples: List[Dict[str, Any]],
    sampling_params: SamplingParams,
    batch_size: int,
    desc: str,
    lora_request: Optional[LoRARequest],
):
    results = {}
    for batch in tqdm(list(batched(examples, batch_size)), desc=desc):
        prompt_token_ids = [item["prompt_token_ids"] for item in batch]
        outputs = llm.generate(
            prompts=None,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request,
            use_tqdm=False,
        )
        for item, output in zip(batch, outputs):
            prompt_logprobs = output.prompt_logprobs or []
            token_ids = list(output.prompt_token_ids or [])
            start = item["prompt_len"]
            stop = min(item["full_len"], len(prompt_logprobs), len(token_ids))
            values = []
            for pos in range(start, stop):
                lp = token_logprob(prompt_logprobs[pos], token_ids[pos])
                if lp is not None and math.isfinite(lp):
                    values.append(lp)
            key = (item["row_index"], item["response_name"])
            if values:
                results[key] = {
                    "logp": float(sum(values)),
                    "tokens": len(values),
                    "truncated": bool(item["truncated"]),
                    "valid": True,
                }
            else:
                results[key] = {
                    "logp": float("nan"),
                    "tokens": 0,
                    "truncated": bool(item["truncated"]),
                    "valid": False,
                }
    return results


def score_source(rows: List[Dict[str, Any]], llm: LLM, tokenizer, sampling_params, policy_lora, ref_lora, args, source_name: str):
    examples, scored, skipped = build_examples(rows, tokenizer, args, source_name)
    if not examples:
        return scored, skipped

    policy = sum_response_logprobs(
        llm,
        examples,
        sampling_params,
        args.batch_size,
        f"Policy {source_name}",
        policy_lora,
    )
    ref = {}
    if args.ref_mode != "none":
        ref = sum_response_logprobs(
            llm,
            examples,
            sampling_params,
            args.batch_size,
            f"Ref {source_name}",
            ref_lora,
        )

    for idx, out in enumerate(scored):
        if (idx, "chosen") not in policy or (idx, "rejected") not in policy:
            continue
        pc = policy.get((idx, "chosen"), {})
        pr = policy.get((idx, "rejected"), {})
        rc = ref.get((idx, "chosen"), {})
        rr = ref.get((idx, "rejected"), {})
        valid_items = (pc, pr) if args.ref_mode == "none" else (pc, pr, rc, rr)
        valid = all(item.get("valid") for item in valid_items)
        if valid:
            policy_logratio = pc["logp"] - pr["logp"]
            ref_logratio = 0.0 if args.ref_mode == "none" else rc["logp"] - rr["logp"]
            margin = args.beta * (policy_logratio - ref_logratio)
            out.update(
                {
                    "client_margin": float(margin),
                    "dpo_margin": float(margin),
                    "client_policy_logratio": float(policy_logratio),
                    "client_ref_logratio": float(ref_logratio),
                    "client_policy_chosen_logp": float(pc["logp"]),
                    "client_policy_rejected_logp": float(pr["logp"]),
                    "client_ref_chosen_logp": float(rc.get("logp", 0.0)),
                    "client_ref_rejected_logp": float(rr.get("logp", 0.0)),
                    "client_margin_valid": True,
                }
            )
        else:
            out["client_margin_valid"] = False
            out["client_margin_error"] = "missing/non-finite prompt logprob"
            skipped += 1
        out.update(
            {
                "client_policy_chosen_tokens": int(pc.get("tokens", 0)),
                "client_policy_rejected_tokens": int(pr.get("tokens", 0)),
                "client_ref_chosen_tokens": int(rc.get("tokens", 0)),
                "client_ref_rejected_tokens": int(rr.get("tokens", 0)),
                "client_truncated": bool(
                    pc.get("truncated") or pr.get("truncated") or rc.get("truncated") or rr.get("truncated")
                ),
            }
        )
    return scored, skipped


def main():
    parser = argparse.ArgumentParser(description="Fast vLLM scorer for client policy-vs-base reference DPO margins.")
    parser.add_argument("--input", action="append", required=True, help="name=/path/file.jsonl; repeatable")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--policy_adapter", type=str, default=None)
    parser.add_argument("--ref_adapter", type=str, default=None,
                        help="Path to reference model LoRA adapter. If None, uses base model.")
    parser.add_argument(
        "--ref_mode",
        choices=("base", "none"),
        default="base",
        help="base: DPO-style policy-vs-base margin; none: score by policy chosen-vs-rejected logratio only.",
    )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--prompt_key", type=str, default="query")
    parser.add_argument("--chosen_key", type=str, default="chosen_response")
    parser.add_argument("--rejected_key", type=str, default="reject_response")
    parser.add_argument("--system_key", type=str, default="system")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument("--max_loras", type=int, default=2)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graph capture to reduce peak memory and improve OOM recovery.",
    )
    parser.add_argument(
        "--max_rows_per_source",
        type=int,
        default=-1,
        help="If > 0, score at most this many rows from each named input source.",
    )
    parser.add_argument(
        "--row_sample_mode",
        choices=("first", "random"),
        default="first",
        help="How to choose rows when --max_rows_per_source is enabled.",
    )
    parser.add_argument("--row_sample_seed", type=int, default=42)
    parser.add_argument("--use_chat_template", action="store_true")
    args = parser.parse_args()

    if args.max_num_batched_tokens < args.max_model_len:
        print(
            "[Warning] max_num_batched_tokens "
            f"({args.max_num_batched_tokens}) is smaller than max_model_len "
            f"({args.max_model_len}); setting max_num_batched_tokens to max_model_len."
        )
        args.max_num_batched_tokens = args.max_model_len

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    enable_lora = bool(args.policy_adapter or args.ref_adapter)
    print(
        f"[Load vLLM] model={args.model}, policy_adapter={args.policy_adapter}, "
        f"ref_adapter={args.ref_adapter}, ref_mode={args.ref_mode}"
    )
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer_path or args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_lora=enable_lora,
        max_lora_rank=args.max_lora_rank,
        max_loras=args.max_loras,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
    )
    policy_lora = LoRARequest("client_policy", 1, args.policy_adapter) if args.policy_adapter else None
    ref_lora = None
    if args.ref_adapter:
        ref_lora = LoRARequest("client_ref", 2, args.ref_adapter)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    summary = {}
    for source_idx, spec in enumerate(args.input):
        name, path = parse_named_path(spec)
        rows = read_jsonl(path)
        num_input_before_limit = len(rows)
        rows = limit_rows(
            rows,
            args.max_rows_per_source,
            args.row_sample_mode,
            args.row_sample_seed + source_idx,
            name,
        )
        scored, skipped = score_source(rows, llm, tokenizer, sampling_params, policy_lora, ref_lora, args, name)
        out_path = os.path.join(args.output_dir, f"{name}.jsonl")
        save_jsonl(scored, out_path)
        valid_margins = [r["client_margin"] for r in scored if r.get("client_margin_valid")]
        summary[name] = {
            "input": path,
            "output": out_path,
            "num_input_before_limit": num_input_before_limit,
            "max_rows_per_source": args.max_rows_per_source,
            "row_sample_mode": args.row_sample_mode,
            "num_input": len(rows),
            "num_output": len(scored),
            "num_invalid": skipped,
            "mean_client_margin": sum(valid_margins) / len(valid_margins) if valid_margins else None,
        }
        print(f"[Done] {name}: {len(scored)} rows -> {out_path}")

    with open(os.path.join(args.output_dir, "score_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
