# encoding=utf-8
import copy
import torch
from openrlhf.trainer import DPOTrainer


def is_lora_a_key(key: str) -> bool:
    return "lora_A" in key


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


def get_fed_local_dpo_trainer(
    model,
    ref_model,
    strategy,
    optim,
    train_dataloader,
    eval_dataloader,
    scheduler,
    max_norm,
    beta,
    max_epochs,
    tokenizer,
    save_hf_ckpt,
    disable_ds_ckpt,
    fed_alg,
    global_dict,
    local_auxiliary,
    global_auxiliary,
    prox_mu,
):
    aggregate_a_only = is_dual_lora_fed_alg(fed_alg)

    if fed_alg == 'fedprox':
        trainer = DPOTrainerFedProx(
            model=model,
            ref_model=ref_model,
            strategy=strategy,
            tokenizer=tokenizer,
            optim=optim,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            scheduler=scheduler,
            max_norm=max_norm,
            beta=beta,
            max_epochs=max_epochs,
            save_hf_ckpt=save_hf_ckpt,
            disable_ds_ckpt=disable_ds_ckpt,
            global_state=global_dict,
            prox_mu=prox_mu,
            aggregate_a_only=aggregate_a_only,
        )
    elif fed_alg == 'scaffold':
        trainer = DPOTrainerSCAFFOLD(
            model=model,
            ref_model=ref_model,
            strategy=strategy,
            tokenizer=tokenizer,
            optim=optim,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            scheduler=scheduler,
            max_norm=max_norm,
            beta=beta,
            max_epochs=max_epochs,
            save_hf_ckpt=save_hf_ckpt,
            disable_ds_ckpt=disable_ds_ckpt,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
            aggregate_a_only=aggregate_a_only,
        )
    elif (
        fed_alg in ['fedavg', 'fedavgm', 'fedadagrad', 'fedadgrad', 'fedyogi', 'fedadam']
    ) or str(fed_alg).startswith('local'):
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            strategy=strategy,
            tokenizer=tokenizer,
            optim=optim,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            scheduler=scheduler,
            max_norm=max_norm,
            beta=beta,
            max_epochs=max_epochs,
            save_hf_ckpt=save_hf_ckpt,
            disable_ds_ckpt=disable_ds_ckpt,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_alg}')
    return trainer


class DPOTrainerFedProx(DPOTrainer):
    def __init__(self, global_state, prox_mu, aggregate_a_only=True, **kwargs):
        super(DPOTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
        self.aggregate_a_only = aggregate_a_only

    def compute_loss(self, model, inputs, return_outputs=False):
        return_values = super(DPOTrainerFedProx, self).compute_loss(
            model, inputs, return_outputs=return_outputs
        )

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        for name, param in model.named_parameters():
            name = name.replace(".default", "")
            if not param.requires_grad:
                continue
            if self.aggregate_a_only and not is_lora_a_key(name):
                continue
            loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class DPOTrainerSCAFFOLD(DPOTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, aggregate_a_only=True, **kwargs):
        super(DPOTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.aggregate_a_only = aggregate_a_only
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            if self.aggregate_a_only and not is_lora_a_key(name):
                self.correction[name] = torch.zeros_like(self.correction[name])
            else:
                self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]

    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                name = name.replace(".default", "")
                if self.aggregate_a_only and not is_lora_a_key(name):
                    auxiliary_new_para[name] = self.local_auxiliary[name]
                    auxiliary_delta_para[name] = torch.zeros_like(self.local_auxiliary[name])
                    continue

                auxiliary_new_para[name] = (
                    (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate)
                    - self.correction[name]
                )
                auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]

        return auxiliary_new_para, auxiliary_delta_para
