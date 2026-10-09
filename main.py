# encoding=utf-8
import argparse
import copy
import json
import math
import os
import gc
import glob
import torch
import safetensors.torch
from datetime import datetime
from transformers.trainer import get_scheduler

# OpenRLHF
from openrlhf.datasets import RewardDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor
from openrlhf.utils import get_strategy, get_tokenizer

# Federated
from federated_learning.fed_local import get_fed_local_dpo_trainer
from federated_learning.fed_global import global_aggregate
from federated_learning.fed_utils import get_auxiliary_dict, get_proxy_dict
from peft import get_peft_model_state_dict, set_peft_model_state_dict


print(f"Rank {os.environ.get('RANK')}: Visible Devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Rank {os.environ.get('RANK')}: Current Device: {torch.cuda.current_device()}")


def state_dict_to_cpu(state_dict):
    """Keep adapter states off GPU; federated aggregation only needs tensors, not CUDA residency."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def align_weights(target_keys, source_dict):
    aligned_dict = {}
    source_keys = list(source_dict.keys())

    def get_significant_suffix(k):
        for marker in ["layers.", "embed_tokens.", "norm.", "lm_head.", "lora_A.", "lora_B."]:
            if marker in k:
                return k[k.find(marker):]
        return k

    source_map = {get_significant_suffix(k): k for k in source_keys}

    for t_k in target_keys:
        if t_k in source_dict:
            aligned_dict[t_k] = source_dict[t_k].detach().cpu().clone()
            continue

        t_suffix = get_significant_suffix(t_k)
        if t_suffix in source_map:
            s_k = source_map[t_suffix]
            aligned_dict[t_k] = source_dict[s_k].detach().cpu().clone()
            continue

        found = False
        for s_k in source_keys:
            if t_k.endswith(s_k) or s_k.endswith(t_k):
                aligned_dict[t_k] = source_dict[s_k].detach().cpu().clone()
                found = True
                break

        if not found:
            print(f"[Warning] Global key not found in local model: {t_k}")

    return aligned_dict


def is_lora_a_key(key: str) -> bool:
    return "lora_A" in key


def is_lora_b_key(key: str) -> bool:
    return "lora_B" in key


def is_dual_lora_fed_alg(fed_alg: str) -> bool:
    if fed_alg is None:
        return False
    fed_alg = str(fed_alg).lower()
    if fed_alg.startswith("local"):
        # Local-only runs still need per-client full A+B inheritance.
        return True
    return fed_alg in {
        "fedavg",
        "fedavgm",
        "fedadagrad",
        "fedadgrad",
        "fedyogi",
        "fedadam",
        "scaffold",
        "fedprox",
    }


def build_dual_lora_client_init_state(global_a_dict, local_personal_dict, prefer_local_a=False):
    merged = copy.deepcopy(local_personal_dict)
    for key in merged.keys():
        if is_lora_a_key(key):
            if prefer_local_a:
                merged[key] = local_personal_dict[key]
            else:
                merged[key] = global_a_dict[key]
        elif is_lora_b_key(key):
            merged[key] = local_personal_dict[key]
        else:
            merged[key] = global_a_dict.get(key, local_personal_dict[key])
    return merged


def build_dual_lora_export_global_state(global_a_dict, local_dict_list, sample_num_list, clients_this_round):
    export_state = copy.deepcopy(global_a_dict)
    if len(clients_this_round) == 0:
        return export_state

    sample_this_round = max(sum([sample_num_list[c] for c in clients_this_round]), 1)

    for key in export_state.keys():
        if is_lora_b_key(key):
            export_state[key] = sum(
                [local_dict_list[c][key] * sample_num_list[c] / sample_this_round for c in clients_this_round]
            )
    return export_state



def build_global_a_local_b_reference_state(global_a_state, local_state):
    updated = copy.deepcopy(local_state)
    num_lora_a = 0
    num_lora_b = 0

    for key, local_value in list(updated.items()):
        if is_lora_a_key(key) and key in global_a_state:
            updated[key] = global_a_state[key].detach().cpu().to(dtype=local_value.dtype).clone()
            num_lora_a += 1
        elif is_lora_b_key(key):
            num_lora_b += 1

    stats = {
        "global_A_source": "post_aggregate_global_A",
        "local_B_source": "client_local_B",
        "num_lora_A_tensors": int(num_lora_a),
        "num_lora_B_tensors": int(num_lora_b),
    }
    return updated, stats


def get_ref_load_in_4bit(args):
    return bool(getattr(args, "ref_load_in_4bit", False))


def cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_reference_model_engine(args, strategy, ref_adapter_state=None):
    base_path = args.pretrain if ref_adapter_state is not None else args.ref_pretrain

    if ref_adapter_state is not None:
        ref_model = Actor(
            base_path,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            load_in_4bit=get_ref_load_in_4bit(args),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
            packing_samples=args.packing_samples,
        )
    else:
        ref_model = Actor(
            base_path,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            load_in_4bit=get_ref_load_in_4bit(args),
            ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
            packing_samples=args.packing_samples,
        )

    if ref_adapter_state is not None:
        set_peft_model_state_dict(ref_model.model, ref_adapter_state)

    for param in ref_model.model.parameters():
        param.requires_grad_(False)

    if args.ref_offload:
        ref_model._offload = True

    return strategy.prepare(ref_model)


def destroy_model_engine(model_engine):
    if model_engine is None:
        return
    if hasattr(model_engine, "destroy"):
        model_engine.destroy()
    elif hasattr(model_engine, "module") and hasattr(model_engine.module, "destroy"):
        model_engine.module.destroy()
    elif hasattr(model_engine, "model") and hasattr(model_engine.model, "destroy"):
        model_engine.model.destroy()


def load_adapter_state_from_dir(client_save_dir, strategy, client_model_engine):
    safe_path = os.path.join(client_save_dir, "adapter_model.safetensors")
    bin_path = os.path.join(client_save_dir, "adapter_model.bin")

    if os.path.exists(safe_path):
        raw_local_dict = safetensors.torch.load_file(safe_path, device="cpu")
    elif os.path.exists(bin_path):
        raw_local_dict = torch.load(bin_path, map_location="cpu")
    else:
        unwrapped_trained_model = strategy._unwrap_model(client_model_engine)
        raw_local_dict = get_peft_model_state_dict(unwrapped_trained_model.model)
        raw_local_dict = {k: v.cpu().clone() for k, v in raw_local_dict.items()}

    return raw_local_dict



def load_adapter_state_from_adapter_dir(adapter_dir: str):
    if adapter_dir is None or str(adapter_dir).strip() == "":
        raise ValueError("adapter_dir is empty")

    safe_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    lora_a_path = os.path.join(adapter_dir, "lora_A.safetensors")
    lora_b_path = os.path.join(adapter_dir, "lora_B.safetensors")

    if os.path.exists(safe_path):
        return safetensors.torch.load_file(safe_path, device="cpu")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")

    state = {}
    if os.path.exists(lora_a_path):
        state.update(safetensors.torch.load_file(lora_a_path, device="cpu"))
    if os.path.exists(lora_b_path):
        state.update(safetensors.torch.load_file(lora_b_path, device="cpu"))
    if state:
        return state

    raise FileNotFoundError(
        f"No adapter weights found under {adapter_dir}. Expected adapter_model.safetensors/bin or lora_A/B.safetensors"
    )


def parse_client_adapter_paths(client_adapter_paths: str):
    mapping = {}
    if client_adapter_paths is None or str(client_adapter_paths).strip() == "":
        return mapping
    for item in client_adapter_paths.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                f"Invalid client adapter entry: '{item}'. Expected format like 0:/abs/path/adapter_dir"
            )
        client_str, path_str = item.split(':', 1)
        mapping[int(client_str.strip())] = path_str.strip()
    return mapping

def to_cpu_tensor_dict(state_dict, key_filter=None):
    out = {}
    for key, value in state_dict.items():
        if key_filter is not None and not key_filter(key):
            continue
        out[key] = value.detach().cpu().clone()
    return out


def safe_save_safetensors(tensor_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    safetensors.torch.save_file(tensor_dict, save_path)


def save_lora_ab_matrices(adapter_state_dict, save_dir, metadata=None, save_full_adapter=True):
    os.makedirs(save_dir, exist_ok=True)

    lora_a_dict = to_cpu_tensor_dict(adapter_state_dict, is_lora_a_key)
    lora_b_dict = to_cpu_tensor_dict(adapter_state_dict, is_lora_b_key)

    safe_save_safetensors(lora_a_dict, os.path.join(save_dir, "lora_A.safetensors"))
    safe_save_safetensors(lora_b_dict, os.path.join(save_dir, "lora_B.safetensors"))

    if save_full_adapter:
        full_adapter_dict = to_cpu_tensor_dict(adapter_state_dict)
        safe_save_safetensors(full_adapter_dict, os.path.join(save_dir, "adapter_model.safetensors"))

    meta = {
        "num_lora_A_tensors": len(lora_a_dict),
        "num_lora_B_tensors": len(lora_b_dict),
        "num_total_tensors": len(adapter_state_dict),
        "files": {
            "lora_A": "lora_A.safetensors",
            "lora_B": "lora_B.safetensors",
            "full_adapter": "adapter_model.safetensors" if save_full_adapter else None,
        },
    }
    if metadata is not None:
        meta.update(metadata)

    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_global_a_state(global_state_dict, save_dir, metadata=None):
    os.makedirs(save_dir, exist_ok=True)
    lora_a_dict = to_cpu_tensor_dict(global_state_dict, is_lora_a_key)
    safe_save_safetensors(lora_a_dict, os.path.join(save_dir, "lora_A.safetensors"))

    meta = {
        "num_lora_A_tensors": len(lora_a_dict),
        "files": {"lora_A": "lora_A.safetensors"},
    }
    if metadata is not None:
        meta.update(metadata)

    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_client_data_paths(client_data_paths: str):
    mapping = {}
    if client_data_paths is None or str(client_data_paths).strip() == "":
        return mapping

    for item in client_data_paths.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                f"Invalid --client_data_paths entry: '{item}'. Expected format like 0:/abs/path/file.jsonl"
            )
        client_str, path_str = item.split(':', 1)
        mapping[int(client_str.strip())] = path_str.strip()
    return mapping


def resolve_client_data_path(args, client_id: int, round_idx: int):
    explicit_map = parse_client_data_paths(args.client_data_paths)
    if client_id in explicit_map:
        candidate = explicit_map[client_id]
        if not os.path.isabs(candidate):
            candidate = os.path.join(args.dataset_prefix, candidate)
        return candidate if os.path.exists(candidate) else None

    iter_id = args.data_iter if args.data_iter is not None else round_idx

    candidates = [
        os.path.join(args.dataset_prefix, f"client_{client_id}", "dpo_train.jsonl"),
        os.path.join(args.dataset_prefix, f"iter_{iter_id}", f"client_{client_id}", "dpo_train.jsonl"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    recursive_patterns = [
        os.path.join(args.dataset_prefix, "**", f"iter_{iter_id}", f"client_{client_id}", "dpo_train.jsonl"),
        os.path.join(args.dataset_prefix, "**", f"client_{client_id}", "dpo_train.jsonl"),
    ]

    matched = []
    for pattern in recursive_patterns:
        matched.extend(glob.glob(pattern, recursive=True))

    matched = sorted(set(matched))
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        print(f"[Warning] Multiple dataset files found for client_{client_id}: {matched}. Use --client_data_paths to disambiguate.")
        return matched[0]
    return None


def train(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    strategy.print(">>> Pre-initializing Models...")

    temp_model = Actor(
        args.pretrain,
        attn_implementation=args.attn_implementation,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
        use_liger_kernel=args.use_liger_kernel,
    )
    global_dict = state_dict_to_cpu(get_peft_model_state_dict(temp_model.model))
    init_global_dict = copy.deepcopy(global_dict)

    local_personal_dict_list = [copy.deepcopy(global_dict) for _ in range(args.num_clients)]
    local_dict_list = [copy.deepcopy(global_dict) for _ in range(args.num_clients)]

    local_reference_dict_list = [None for _ in range(args.num_clients)]

    if args.client_ids is not None:
        active_clients = [int(x.strip()) for x in args.client_ids.split(",")]
    else:
        active_clients = list(range(args.num_clients))
    strategy.print(f">>> Active Clients for this run: {active_clients}")

    client_init_adapter_map = parse_client_adapter_paths(getattr(args, "client_init_adapter_paths", None))
    client_ref_adapter_map = parse_client_adapter_paths(getattr(args, "client_ref_adapter_paths", None))
    external_init_state_dicts = {}
    external_ref_state_dicts = {}

    for cid, adapter_dir in client_init_adapter_map.items():
        if cid < 0 or cid >= args.num_clients:
            raise ValueError(f"client_init_adapter_paths has invalid client id {cid}")
        strategy.print(f"[External Init] client_{cid} policy init adapter: {adapter_dir}")
        raw_state = load_adapter_state_from_adapter_dir(adapter_dir)
        aligned_state = align_weights(global_dict.keys(), raw_state)
        external_init_state_dicts[cid] = aligned_state
        local_personal_dict_list[cid] = copy.deepcopy(aligned_state)
        local_dict_list[cid] = copy.deepcopy(aligned_state)

    for cid, adapter_dir in client_ref_adapter_map.items():
        if cid < 0 or cid >= args.num_clients:
            raise ValueError(f"client_ref_adapter_paths has invalid client id {cid}")
        strategy.print(f"[External Ref] client_{cid} DPO reference adapter: {adapter_dir}")
        raw_state = load_adapter_state_from_adapter_dir(adapter_dir)
        external_ref_state_dicts[cid] = align_weights(global_dict.keys(), raw_state)

    proxy_dict, opt_proxy_dict = get_proxy_dict(args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(args, global_dict)


    use_external_ref_for_all_active = (
        len(client_ref_adapter_map) > 0
        and all(cid in client_ref_adapter_map for cid in active_clients)
    )

    if use_external_ref_for_all_active:
        tokenizer = get_tokenizer(
            args.pretrain,
            temp_model.model,
            "right",
            strategy,
            use_fast=not args.disable_fast_tokenizer,
        )
        ref_model_engine = None
        strategy.print(">>> Using disk-provided per-client reference adapters; skip persistent base reference model.")
    else:
        ref_model = Actor(
            args.ref_pretrain,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            load_in_4bit=get_ref_load_in_4bit(args),
            ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
            packing_samples=args.packing_samples,
        )
        if args.ref_offload:
            ref_model._offload = True
        for param in ref_model.model.parameters():
            param.requires_grad_(False)
        ref_model_engine = strategy.prepare(ref_model)

        tokenizer = get_tokenizer(
            args.pretrain,
            ref_model.model,
            "right",
            strategy,
            use_fast=not args.disable_fast_tokenizer,
        )

    del temp_model
    cleanup_cuda_memory()

    for round_idx in range(args.rounds):
        strategy.print(f"\n[Round {round_idx + 1}/{args.rounds}] Global Training Start")

        if args.iterative_ref_update and round_idx > 0 and ref_model_engine is not None:
            destroy_model_engine(ref_model_engine)
            ref_model_engine = None
            cleanup_cuda_memory()

        sample_num_list = [0] * args.num_clients
        active_clients_list = []

        for client_id in active_clients:
            client_model_base = Actor(
                args.pretrain,
                attn_implementation=args.attn_implementation,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules,
                ds_config=strategy.get_ds_train_config(is_actor=True),
                packing_samples=args.packing_samples,
                use_liger_kernel=args.use_liger_kernel,
            )

            if is_dual_lora_fed_alg(args.fed_alg):
                init_state = build_dual_lora_client_init_state(
                    global_a_dict=global_dict,
                    local_personal_dict=local_personal_dict_list[client_id],
                    prefer_local_a=(
                        client_id in external_init_state_dicts
                        or (
                            args.iterative_ref_update
                            and round_idx > 0
                            and local_reference_dict_list[client_id] is not None
                        )
                    ),
                )
                set_peft_model_state_dict(client_model_base.model, init_state)
            else:
                set_peft_model_state_dict(client_model_base.model, global_dict)

            if args.gradient_checkpointing:
                client_model_base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": args.gradient_checkpointing_use_reentrant
                    }
                )

            optim = strategy.create_optimizer(
                client_model_base,
                lr=args.learning_rate,
                betas=args.adam_betas,
                weight_decay=args.l2,
            )

            strategy.print(f"--- Client {client_id} Training ---")
            client_data_path = resolve_client_data_path(args, client_id, round_idx)

            if not client_data_path or not os.path.exists(client_data_path):
                strategy.print(f"[Warning] Client {client_id} data not found, skipping. dataset_prefix={args.dataset_prefix}")
                del client_model_base, optim
                continue

            strategy.print(f"[Client {client_id}] Loading data from: {client_data_path}")

            train_data = blending_datasets(
                client_data_path,
                args.dataset_probs,
                strategy,
                args.seed,
                max_count=args.max_samples,
                dataset_split=args.dataset_split,
            )
            train_data = train_data.select(range(min(args.max_samples, len(train_data))))

            train_dataset = RewardDataset(
                train_data,
                tokenizer,
                args.max_len,
                strategy,
                input_template=args.input_template,
                is_dpo=True,
            )

            sample_num_list[client_id] = len(train_dataset)
            active_clients_list.append(client_id)

            train_dataloader = strategy.setup_dataloader(
                train_dataset,
                args.micro_train_batch_size,
                True,
                True,
                train_dataset.collate_fn,
            )

            eval_dataset = None
            eval_dataloader = None
            if getattr(args, "eval_dataset", None):
                eval_data = blending_datasets(
                    args.eval_dataset,
                    None,
                    strategy,
                    dataset_split=args.eval_split,
                )
                eval_dataset = RewardDataset(
                    eval_data,
                    tokenizer,
                    args.max_len,
                    strategy,
                    input_template=args.input_template,
                    is_dpo=True,
                )
                eval_dataloader = strategy.setup_dataloader(
                    eval_dataset,
                    args.micro_train_batch_size,
                    True,
                    False,
                    eval_dataset.collate_fn,
                )

            num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
            num_update_steps_per_epoch = max(1, num_update_steps_per_epoch)
            max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

            scheduler = get_scheduler(
                args.lr_scheduler,
                optim,
                num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr": args.learning_rate * args.min_lr_ratio},
            )

            client_model_engine, optim, scheduler = strategy.prepare(
                (client_model_base, optim, scheduler)
            )

            client_ref_model_engine = ref_model_engine
            use_personalized_ref_engine = False
            if client_id in external_ref_state_dicts:
                strategy.print(
                    f"[Client {client_id}] Using disk-provided personalized DPO reference adapter."
                )
                cleanup_cuda_memory()
                client_ref_model_engine = build_reference_model_engine(
                    args=args,
                    strategy=strategy,
                    ref_adapter_state=external_ref_state_dicts[client_id],
                )
                use_personalized_ref_engine = True
            elif (
                args.iterative_ref_update
                and round_idx > 0
                and local_reference_dict_list[client_id] is not None
            ):
                strategy.print(
                    f"[Client {client_id}] Using personalized iterative DPO reference model "
                    f"from previous round."
                )
                cleanup_cuda_memory()
                client_ref_model_engine = build_reference_model_engine(
                    args=args,
                    strategy=strategy,
                    ref_adapter_state=local_reference_dict_list[client_id],
                )
                use_personalized_ref_engine = True
            elif client_ref_model_engine is None:
                client_ref_model_engine = build_reference_model_engine(
                    args=args,
                    strategy=strategy,
                    ref_adapter_state=None,
                )
                use_personalized_ref_engine = True

            trainer = get_fed_local_dpo_trainer(
                model=client_model_engine,
                ref_model=client_ref_model_engine,
                strategy=strategy,
                optim=optim,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                scheduler=scheduler,
                max_norm=args.max_norm,
                beta=args.beta,
                max_epochs=args.max_epochs,
                tokenizer=tokenizer,
                save_hf_ckpt=args.save_hf_ckpt,
                disable_ds_ckpt=args.disable_ds_ckpt,
                fed_alg=args.fed_alg,
                global_dict=global_dict,
                local_auxiliary=auxiliary_model_list[client_id],
                global_auxiliary=global_auxiliary,
                prox_mu=args.prox_mu,
            )

            train_stats = trainer.fit(args, num_update_steps_per_epoch=num_update_steps_per_epoch)
            if train_stats is None and hasattr(trainer, "get_train_stats"):
                train_stats = trainer.get_train_stats()
            if train_stats is None:
                train_stats = getattr(trainer, "train_stats", {})
            if train_stats is None:
                train_stats = {}
            train_stats = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in dict(train_stats).items()}
            train_stats["client_id"] = int(client_id)
            train_stats["round"] = int(round_idx + 1)
            train_stats["sample_num"] = int(sample_num_list[client_id])
            strategy.print(f"[Client {client_id}] training-only stats for curriculum selection: {train_stats}")

            client_save_dir = os.path.join(
                args.save_path,
                f"round_{round_idx + 1}",
                f"client_{client_id}"
            )
            strategy.print(f">>> Saving Client {client_id} LoRA to {client_save_dir}")
            strategy.save_model(client_model_engine, tokenizer, client_save_dir)

            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()

            if args.fed_alg == 'scaffold':
                auxiliary_model_list[client_id], auxiliary_delta_dict[client_id] = trainer.get_auxiliary_param()

            raw_local_dict = load_adapter_state_from_dir(client_save_dir, strategy, client_model_engine)
            aligned_local_dict = align_weights(global_dict.keys(), raw_local_dict)

            if strategy.is_rank_0():
                client_ab_dir = os.path.join(
                    args.save_path,
                    f"round_{round_idx + 1}",
                    "client_lora_states",
                    f"client_{client_id}",
                )
                save_lora_ab_matrices(
                    aligned_local_dict,
                    client_ab_dir,
                    metadata={
                        "round": round_idx + 1,
                        "client_id": client_id,
                        "sample_num": sample_num_list[client_id],
                        "fed_alg": args.fed_alg,
                        "state_type": "client_trained_before_server_aggregation",
                        "source_adapter_dir": client_save_dir,
                        "train_stats": train_stats,
                    },
                    save_full_adapter=True,
                )

                save_lora_ab_matrices(
                    aligned_local_dict,
                    os.path.join(client_save_dir, "split_lora"),
                    metadata={
                        "round": round_idx + 1,
                        "client_id": client_id,
                        "sample_num": sample_num_list[client_id],
                        "fed_alg": args.fed_alg,
                        "state_type": "client_trained_before_server_aggregation",
                        "train_stats": train_stats,
                    },
                    save_full_adapter=False,
                )
                # Save training-only metrics separately for auditability / reproducibility.
                train_stats_path = os.path.join(client_ab_dir, "train_stats.json")
                with open(train_stats_path, "w", encoding="utf-8") as f:
                    json.dump(train_stats, f, ensure_ascii=False, indent=2)
                strategy.print(f">>> Saved split LoRA A/B for Client {client_id} to {client_ab_dir}")

            local_dict_list[client_id] = aligned_local_dict

            if is_dual_lora_fed_alg(args.fed_alg):
                local_personal_dict_list[client_id] = copy.deepcopy(aligned_local_dict)

            if hasattr(client_model_engine, "destroy"):
                client_model_engine.destroy()
            elif hasattr(client_model_engine, "module") and hasattr(client_model_engine.module, "destroy"):
                client_model_engine.module.destroy()
            elif hasattr(client_model_engine, "model") and hasattr(client_model_engine.model, "destroy"):
                client_model_engine.model.destroy()

            if use_personalized_ref_engine:
                destroy_model_engine(client_ref_model_engine)
                cleanup_cuda_memory()

            del client_model_engine, optim, scheduler, train_dataloader, train_dataset, trainer, client_model_base
            if use_personalized_ref_engine:
                del client_ref_model_engine
            if 'train_data' in locals():
                del train_data
            if 'eval_dataset' in locals() and eval_dataset is not None:
                del eval_dataset, eval_dataloader
            if 'raw_local_dict' in locals():
                del raw_local_dict
            if 'aligned_local_dict' in locals():
                del aligned_local_dict

            cleanup_cuda_memory()

        if args.fed_alg == "scaffold":
            auxiliary_info = (global_auxiliary, auxiliary_delta_dict)
        else:
            auxiliary_info = None

        if strategy.is_rank_0():
            round_dir = os.path.join(args.save_path, f"round_{round_idx + 1}")
            save_global_a_state(
                global_dict,
                os.path.join(round_dir, "pre_aggregate_global_A"),
                metadata={
                    "round": round_idx + 1,
                    "fed_alg": args.fed_alg,
                    "state_type": "server_global_A_before_aggregation",
                    "active_clients": active_clients_list,
                },
            )

        skip_global_aggregation = str(args.fed_alg).lower().startswith("local")
        if skip_global_aggregation:
            strategy.print(
                f"[Round {round_idx + 1}] Local-only training enabled "
                f"(fed_alg={args.fed_alg}); skip server aggregation/upload."
            )
            global_auxiliary_out = None
        else:
            global_dict, global_auxiliary_out = global_aggregate(
                args,
                global_dict,
                local_dict_list,
                sample_num_list,
                active_clients_list,
                round_idx,
                proxy_dict=proxy_dict,
                opt_proxy_dict=opt_proxy_dict,
                auxiliary_info=auxiliary_info,
            )

        if args.fed_alg == "scaffold":
            global_auxiliary = global_auxiliary_out

        iterative_ref_stats = {}
        if args.iterative_ref_update and is_dual_lora_fed_alg(args.fed_alg):
            for cid in active_clients_list:
                updated_personal_state, ref_stats = build_global_a_local_b_reference_state(
                    global_a_state=global_dict,
                    local_state=local_personal_dict_list[cid],
                )
                local_personal_dict_list[cid] = updated_personal_state
                local_reference_dict_list[cid] = copy.deepcopy(updated_personal_state)
                iterative_ref_stats[cid] = ref_stats
                strategy.print(
                    f"[Round {round_idx + 1}] Client {cid} iterative ref update: "
                    f"A=post_aggregate_global_A, B=local_B, "
                    f"num_A={ref_stats['num_lora_A_tensors']}, "
                    f"num_B={ref_stats['num_lora_B_tensors']}"
                )

        if strategy.is_rank_0():
            strategy.print(f">>> Saving Round {round_idx + 1} Models")
            round_dir = os.path.join(args.save_path, f"round_{round_idx + 1}")
            os.makedirs(round_dir, exist_ok=True)

            save_global_a_state(
                global_dict,
                os.path.join(round_dir, "post_aggregate_global_A"),
                metadata={
                    "round": round_idx + 1,
                    "fed_alg": args.fed_alg,
                    "state_type": "server_global_A_after_aggregation",
                    "active_clients": active_clients_list,
                },
            )

            if args.skip_export_global_model:
                strategy.print("[Save] skip export-only global_model (--skip_export_global_model).")
            else:
                if is_dual_lora_fed_alg(args.fed_alg):
                    export_state = build_dual_lora_export_global_state(
                        global_a_dict=global_dict,
                        local_dict_list=local_dict_list,
                        sample_num_list=sample_num_list,
                        clients_this_round=active_clients_list,
                    )
                else:
                    export_state = global_dict

                save_dir = os.path.join(round_dir, "global_model")
                save_model = Actor(
                    args.pretrain,
                    attn_implementation=args.attn_implementation,
                    bf16=args.bf16,
                    load_in_4bit=args.load_in_4bit,
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=args.target_modules,
                    ds_config=None,
                )
                set_peft_model_state_dict(save_model.model, export_state)
                save_model.model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                del save_model
                cleanup_cuda_memory()

            if args.skip_global_share_a_only:
                strategy.print("[Save] skip global_share_a_only (--skip_global_share_a_only).")
            elif is_dual_lora_fed_alg(args.fed_alg):
                share_a_dir = os.path.join(round_dir, "global_share_a_only")
                share_a_model = Actor(
                    args.pretrain,
                    attn_implementation=args.attn_implementation,
                    bf16=args.bf16,
                    load_in_4bit=args.load_in_4bit,
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=args.target_modules,
                    ds_config=None,
                )

                share_a_state = copy.deepcopy(init_global_dict)
                for key in share_a_state.keys():
                    if is_lora_a_key(key):
                        share_a_state[key] = global_dict[key]
                set_peft_model_state_dict(share_a_model.model, share_a_state)
                share_a_model.model.save_pretrained(share_a_dir)
                tokenizer.save_pretrained(share_a_dir)
                del share_a_model
                cleanup_cuda_memory()

            if args.iterative_ref_update and is_dual_lora_fed_alg(args.fed_alg):
                iterative_ref_root = os.path.join(round_dir, "iterative_reference_adapters")
                os.makedirs(iterative_ref_root, exist_ok=True)
                for cid in active_clients_list:
                    ref_dir = os.path.join(iterative_ref_root, f"client_{cid}")
                    save_lora_ab_matrices(
                        local_reference_dict_list[cid],
                        ref_dir,
                        metadata={
                            "round": round_idx + 1,
                            "client_id": cid,
                            "fed_alg": args.fed_alg,
                            "state_type": "next_round_personalized_dpo_reference",
                            "formula": "A_next = post_aggregate_global_A; B_next = local_B",
                            **iterative_ref_stats.get(cid, {}),
                        },
                        save_full_adapter=True,
                    )
                    tokenizer.save_pretrained(ref_dir)

    destroy_model_engine(ref_model_engine)
    strategy.print(">>> Federated Learning Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fed_alg", type=str, default="fedavg", help="the algorithm to use")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dataset_prefix", type=str, required=True, help="Data root dir")
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--fedavg_uniform", action="store_true", default=False,
                        help="Use equal client weights for FedAvg-style aggregation instead of sample-count weights.")
    parser.add_argument("--prox_mu", type=float, default=0.01, help="the mu parameter of FedProx")
    parser.add_argument("--fedopt_tau", type=float, default=1e-3,
                        help="the tau parameter of FedAdagrad, FedYogi and FedAdam")
    parser.add_argument("--fedopt_eta", type=float, default=1e-3,
                        help="the global learning rate parameter of FedAdagrad, FedYogi and FedAdam")
    parser.add_argument("--fedopt_beta1", type=float, default=0.9,
                        help="the beta1 parameter of FedYogi and FedAdam")
    parser.add_argument("--fedopt_beta2", type=float, default=0.99,
                        help="the beta2 parameter of FedYogi and FedAdam")
    parser.add_argument("--save_model_freq", type=int, default=50,
                        help="the frequency to save the model (save every N rounds)")
    parser.add_argument("--dft_alpha", type=float, default=0.7,
                        help="the dft_alpha parameter")
    parser.add_argument("--client_ids", type=str, default=None,
                        help="Comma separated client IDs, e.g., '0,1,2,3'")

    # Iterative DPO reference update
    parser.add_argument("--iterative_ref_update", action="store_true", default=False,
                        help="Enable iterative DPO: after each round, build per-client reference adapters from post-aggregation global A and local B.")
    # Deprecated compatibility args: kept so old shell scripts will not fail, but ignored.
    parser.add_argument("--iter_ref_min_global_weight", type=float, default=0.0,
                        help="Deprecated and ignored. Cosine-based global/local A fusion has been removed.")
    parser.add_argument("--iter_ref_max_global_weight", type=float, default=1.0,
                        help="Deprecated and ignored. Cosine-based global/local A fusion has been removed.")
    parser.add_argument("--iter_ref_cosine_power", type=float, default=1.0,
                        help="Deprecated and ignored. Cosine-based global/local A fusion has been removed.")

    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--skip_export_global_model", action="store_true", default=False,
                        help="Skip saving round_X/global_model. Useful for personalized dual-LoRA runs when disk is tight.")
    parser.add_argument("--skip_global_share_a_only", action="store_true", default=False,
                        help="Skip saving round_X/global_share_a_only. Next-iteration adapter building uses post_aggregate_global_A directly.")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_determinism", action="store_true", default=False,
                        help="Enable reproducible behavior during distributed training")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--ref_load_in_4bit", action="store_true", default=False,
                        help="Load only the DPO reference model in 4bit to reduce memory; policy model is unchanged.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        help="Attention implementation")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # DPO
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Ratio of the minimum learning rate to the initial learning rate.")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kd_coef", type=float, default=0.4)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--ipo", action="store_true", default=False)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--nll_loss_coef", type=float, default=0,
                        help="Regularization with NLL loss, see LLama 3.1 tech report.")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument("--ring_head_stride", type=int, default=1,
                        help="the number of heads to do ring attention each time.")

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=20000, help="Maximum number of samples to use")

    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False,
                        help="Use HF tokenizer chat template")
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--client_data_paths", type=str, default=None,
                        help="Optional explicit mapping like 0:/abs/path/client0.jsonl,1:/abs/path/client1.jsonl")
    parser.add_argument("--client_init_adapter_paths", type=str, default=None,
                        help="Outer-shell iteration: mapping like 0:/path/adapter0,1:/path/adapter1. Used to initialize each client policy adapter from disk.")
    parser.add_argument("--client_ref_adapter_paths", type=str, default=None,
                        help="Outer-shell iteration: mapping like 0:/path/adapter0,1:/path/adapter1. Used as each client's DPO reference adapter from disk.")
    parser.add_argument("--data_iter", type=int, default=0,
                        help="Dataset iteration id used in paths like iter_0/client_x when auto resolving data")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument("--wandb_run_name", type=str,
                        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"))

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.packing_samples and "flash_attention" not in args.attn_implementation:
        print(
            "[Warning] Please use --attn_implementation with flash_attention to accelerate when "
            "--packing_samples is enabled."
        )
        args.attn_implementation = "flash_attention_2"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub
        patch_hub()

    train(args)
