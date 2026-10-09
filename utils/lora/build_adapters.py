import argparse
import json
import os
import shutil
from typing import Dict, List

import torch
import safetensors.torch

ADAPTER_SAFE = "adapter_model.safetensors"
ADAPTER_BIN = "adapter_model.bin"
LORA_A_SAFE = "lora_A.safetensors"
LORA_B_SAFE = "lora_B.safetensors"
META_FILENAMES = [
    "adapter_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
]


def is_lora_a_key(k: str) -> bool:
    return "lora_A" in k


def is_lora_b_key(k: str) -> bool:
    return "lora_B" in k


def parse_client_ids(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_adapter_state(adapter_dir: str) -> Dict[str, torch.Tensor]:
    safe_path = os.path.join(adapter_dir, ADAPTER_SAFE)
    bin_path = os.path.join(adapter_dir, ADAPTER_BIN)
    lora_a_path = os.path.join(adapter_dir, LORA_A_SAFE)
    lora_b_path = os.path.join(adapter_dir, LORA_B_SAFE)

    if os.path.exists(safe_path):
        print(f"[Load] {safe_path}")
        return safetensors.torch.load_file(safe_path, device="cpu")
    if os.path.exists(bin_path):
        print(f"[Load] {bin_path}")
        return torch.load(bin_path, map_location="cpu")

    state: Dict[str, torch.Tensor] = {}
    if os.path.exists(lora_a_path):
        print(f"[Load] {lora_a_path}")
        state.update(safetensors.torch.load_file(lora_a_path, device="cpu"))
    if os.path.exists(lora_b_path):
        print(f"[Load] {lora_b_path}")
        state.update(safetensors.torch.load_file(lora_b_path, device="cpu"))
    if state:
        return state

    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def save_adapter_state(state: Dict[str, torch.Tensor], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cpu_state = {k: v.detach().cpu().clone() for k, v in state.items()}
    safetensors.torch.save_file(cpu_state, os.path.join(output_dir, ADAPTER_SAFE))
    lora_a = {k: v for k, v in cpu_state.items() if is_lora_a_key(k)}
    lora_b = {k: v for k, v in cpu_state.items() if is_lora_b_key(k)}
    safetensors.torch.save_file(lora_a, os.path.join(output_dir, LORA_A_SAFE))
    safetensors.torch.save_file(lora_b, os.path.join(output_dir, LORA_B_SAFE))


def copy_metadata(search_dirs: List[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in META_FILENAMES:
        for d in search_dirs:
            src = os.path.join(d, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, name))
                break


def compute_lora_a_cosine(global_a_state: Dict[str, torch.Tensor], local_state: Dict[str, torch.Tensor], eps=1e-12):
    dot = torch.tensor(0.0, dtype=torch.float64)
    gn = torch.tensor(0.0, dtype=torch.float64)
    ln = torch.tensor(0.0, dtype=torch.float64)
    n = 0
    for k, gv in global_a_state.items():
        if not is_lora_a_key(k) or k not in local_state:
            continue
        g = gv.detach().cpu().float().reshape(-1).double()
        l = local_state[k].detach().cpu().float().reshape(-1).double()
        dot += torch.dot(g, l)
        gn += torch.dot(g, g)
        ln += torch.dot(l, l)
        n += 1
    if n == 0:
        return 0.0, 0
    cos = (dot / torch.sqrt(gn * ln).clamp_min(eps)).clamp(-1.0, 1.0).item()
    return float(cos), n


def fuse_global_a_local_b(global_a_state, local_state):
    out = {k: v.detach().cpu().clone() for k, v in local_state.items()}

    replaced_a = 0
    kept_b = 0

    for k, lv in list(out.items()):
        if is_lora_a_key(k) and k in global_a_state:
            out[k] = global_a_state[k].detach().cpu().clone().to(dtype=lv.dtype)
            replaced_a += 1
        elif is_lora_b_key(k):
            kept_b += 1

    return out, {
        "formula": "A_next=global_A; B_next=local_B",
        "replaced_lora_A_tensors": replaced_a,
        "kept_lora_B_tensors": kept_b,
        "use_local_cos": False,
    }


def keep_local_full_adapter(local_state):
    out = {k: v.detach().cpu().clone() for k, v in local_state.items()}
    kept_a = sum(1 for k in out if is_lora_a_key(k))
    kept_b = sum(1 for k in out if is_lora_b_key(k))
    return out, {
        "formula": "A_next=local_A; B_next=local_B",
        "kept_lora_A_tensors": kept_a,
        "kept_lora_B_tensors": kept_b,
        "use_local_full_adapter": True,
    }


def keep_global_full_adapter(global_state):
    out = {k: v.detach().cpu().clone() for k, v in global_state.items()}
    kept_a = sum(1 for k in out if is_lora_a_key(k))
    kept_b = sum(1 for k in out if is_lora_b_key(k))
    return out, {
        "formula": "A_next=global_A; B_next=global_B",
        "kept_lora_A_tensors": kept_a,
        "kept_lora_B_tensors": kept_b,
        "use_global_full_adapter": True,
    }


def find_client_dir(round_dir: str, cid: int) -> str:
    candidates = [
        os.path.join(round_dir, "client_lora_states", f"client_{cid}"),
        os.path.join(round_dir, f"client_{cid}", "split_lora"),
        os.path.join(round_dir, f"client_{cid}"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round_dir", required=True, help=".../iter_k_output/round_1")
    ap.add_argument("--client_ids", default="0,1,2")
    ap.add_argument("--output_dir", required=True, help="Output root for next-round adapters")
    ap.add_argument("--global_a_dir", default=None, help="Default: round_dir/post_aggregate_global_A")
    ap.add_argument(
        "--use_local_full_adapter",
        action="store_true",
        help="Keep each client's trained local A+B instead of replacing A with global A.",
    )
    ap.add_argument(
        "--use_global_full_adapter",
        action="store_true",
        help="Use the round-level full global adapter A+B for every client.",
    )
    args = ap.parse_args()

    if args.use_local_full_adapter and args.use_global_full_adapter:
        raise ValueError("--use_local_full_adapter and --use_global_full_adapter are mutually exclusive")

    client_ids = parse_client_ids(args.client_ids)
    global_a_dir = args.global_a_dir or os.path.join(args.round_dir, "post_aggregate_global_A")
    global_full_dir = os.path.join(args.round_dir, "global_model")
    global_a_state = None
    global_full_state = None
    if args.use_global_full_adapter:
        global_full_state = load_adapter_state(global_full_dir)
    elif not args.use_local_full_adapter:
        global_a_state = load_adapter_state(global_a_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    mapping_parts = []
    all_stats = {}

    for cid in client_ids:
        client_dir = find_client_dir(args.round_dir, cid)
        local_state = load_adapter_state(client_dir)
        if args.use_global_full_adapter:
            next_state, stats = keep_global_full_adapter(global_full_state)
        elif args.use_local_full_adapter:
            next_state, stats = keep_local_full_adapter(local_state)
        else:
            next_state, stats = fuse_global_a_local_b(
                global_a_state=global_a_state,
                local_state=local_state,
            )
        out_dir = os.path.join(args.output_dir, f"client_{cid}")
        save_adapter_state(next_state, out_dir)
        copy_metadata([
            client_dir,
            os.path.join(args.round_dir, f"client_{cid}"),
            os.path.join(args.round_dir, "global_share_a_only"),
            os.path.join(args.round_dir, "global_model"),
            global_a_dir,
        ], out_dir)
        info = {
            "client_id": cid,
            "round_dir": args.round_dir,
            "global_a_dir": global_a_dir,
            "global_full_dir": global_full_dir if args.use_global_full_adapter else None,
            "client_dir": client_dir,
            "formula": stats.get("formula", "A_next=global_A; B_next=local_B"),
            **stats,
        }
        with open(os.path.join(out_dir, "iter_update_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        all_stats[str(cid)] = info
        mapping_parts.append(f"{cid}:{out_dir}")
        print(
            f"[Done] client_{cid}: "
            f"{stats.get('formula')}, "
            f"replaced_A={stats.get('replaced_lora_A_tensors', 0)}, "
            f"kept_A={stats.get('kept_lora_A_tensors', 0)}, "
            f"kept_B={stats['kept_lora_B_tensors']} -> {out_dir}"
        )

    mapping = ",".join(mapping_parts)
    with open(os.path.join(args.output_dir, "client_adapter_paths.txt"), "w", encoding="utf-8") as f:
        f.write(mapping + "\n")
    with open(os.path.join(args.output_dir, "all_iter_update_info.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print("[Mapping]", mapping)


if __name__ == "__main__":
    main()
