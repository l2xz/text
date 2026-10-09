# encoding=utf-8
import json
import math
import os
import torch


def is_lora_a_key(key: str) -> bool:
    return "lora_A" in key


def is_lora_b_key(key: str) -> bool:
    return "lora_B" in key


def is_dual_lora_fed_alg(fed_alg: str) -> bool:
    if fed_alg is None:
        return False
    fed_alg = str(fed_alg).lower()
    if fed_alg.startswith("local"):
        return False
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


def get_aggregate_keys(fed_alg: str, global_dict):
    if is_dual_lora_fed_alg(fed_alg):
        return [key for key in global_dict.keys() if is_lora_a_key(key)]
    return list(global_dict.keys())


def _sample_weights(sample_num_list, clients_this_round):
    sample_this_round = sum([sample_num_list[c] for c in clients_this_round])
    sample_this_round = max(sample_this_round, 1)
    return [sample_num_list[c] / sample_this_round for c in clients_this_round]


def _aggregation_weights(fed_args, sample_num_list, clients_this_round):
    if len(clients_this_round) == 0:
        return {}
    if getattr(fed_args, "fedavg_uniform", False):
        return {client: 1.0 / len(clients_this_round) for client in clients_this_round}

    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    sample_this_round = max(sample_this_round, 1)
    return {client: sample_num_list[client] / sample_this_round for client in clients_this_round}

def global_aggregate(
    fed_args,
    global_dict,
    local_dict_list,
    sample_num_list,
    clients_this_round,
    round_idx,
    margin_list=None,
    kl_list=None,
    proxy_dict=None,
    opt_proxy_dict=None,
    auxiliary_info=None,
):
    if len(clients_this_round) == 0:
        return global_dict, None

    weights = _aggregation_weights(fed_args, sample_num_list, clients_this_round)
    global_auxiliary = None
    agg_keys = get_aggregate_keys(fed_args.fed_alg, global_dict)

    if fed_args.fed_alg == 'scaffold':
        for key in agg_keys:
            global_dict[key] = sum(
                [local_dict_list[client][key] * weights[client] for client in clients_this_round]
            )
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in agg_keys:
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round])
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients

    elif fed_args.fed_alg == 'fedavgm':
        for key in agg_keys:
            delta_w = sum(
                [(local_dict_list[client][key] - global_dict[key]) * weights[client] for client in clients_this_round]
            )
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg in ['fedadagrad', 'fedadgrad']:
        for key in agg_keys:
            param = opt_proxy_dict[key]
            delta_w = sum(
                [(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]
            ) / len(clients_this_round)
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key],
                torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau,
            )

    elif fed_args.fed_alg == 'fedyogi':
        for key in agg_keys:
            param = opt_proxy_dict[key]
            delta_w = sum(
                [(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]
            ) / len(clients_this_round)
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1 - fed_args.fedopt_beta2) * delta_square * torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key],
                torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau,
            )

    elif fed_args.fed_alg == 'fedadam':
        for key in agg_keys:
            param = opt_proxy_dict[key]
            delta_w = sum(
                [(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]
            ) / len(clients_this_round)
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            opt_proxy_dict[key] = fed_args.fedopt_beta2 * param + (1 - fed_args.fedopt_beta2) * torch.square(
                proxy_dict[key]
            )
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key],
                torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau,
            )
    else:
        # FedAvg/FedProx style aggregation. In dual-LoRA mode agg_keys only contains LoRA-A.
        for key in agg_keys:
            global_dict[key] = sum(
                [local_dict_list[client][key] * weights[client] for client in clients_this_round]
            )

    return global_dict, global_auxiliary
