#!/bin/bash
set -e
set -x

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT=${PROJECT_ROOT:-/path/to/FEDCARE}
POLICY_MODEL_PATH=${POLICY_MODEL_PATH:-/path/to/base_model}
HETERO_DATA_ROOT=${ROUND1_DATA_ROOT:-${PROJECT_ROOT}/runs/ours/data/r1}
source "${PROJECT_ROOT}/train_math/common.sh"

MAIN_PY="${PROJECT_ROOT}/main.py"
OFFLINE_ITER_PY="${PROJECT_ROOT}/utils/lora/build_adapters.py"

CLIENT_IDS=${CLIENT_IDS:-0,1,2,3,4,5,6}
CLIENT_IDS_LIST=${CLIENT_IDS//,/ }
NUM_CLIENTS=${NUM_CLIENTS:-7}
TRAINING_ITERS=${TRAINING_ITERS:-1}
DATA_ITER=${DATA_ITER:-0}
SAVE_ROOT=${SAVE_ROOT:-${PROJECT_ROOT}/runs/ours/adapters}
DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-localhost:4,5,6,7}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29510}

# DPO hyperparameters. NLL regularization is disabled by default; override
# NLL regularization is disabled by default for the main experiment.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-1}
MAX_EPOCHS=${MAX_EPOCHS:-1}
MAX_LEN=${MAX_LEN:-1024}
MAX_SAMPLES=${MAX_SAMPLES:-20000}
LEARNING_RATE=${LEARNING_RATE:-2e-6}
BETA=${BETA:-0.1}
NLL_LOSS_COEF=${NLL_LOSS_COEF:-0.0}
FED_ALG=${FED_ALG:-fedavg}
REF_LOAD_IN_4BIT=${REF_LOAD_IN_4BIT:-0}

mkdir -p "${SAVE_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
POLICY_MODEL_PATH="$(resolve_model_path "${POLICY_MODEL_PATH}")"
ensure_deepspeed_batch_size

build_client_data_paths() {
  local data_root="$1"
  local out=""
  local cid
  for cid in ${CLIENT_IDS_LIST}; do
    local p="${data_root}/client_${cid}/dpo_train.jsonl"
    if [[ ! -f "${p}" ]]; then
      echo "[ERROR] Missing DPO data: ${p}" >&2
      exit 1
    fi
    if [[ -z "${out}" ]]; then
      out="${cid}:${p}"
    else
      out="${out},${cid}:${p}"
    fi
  done
  echo "${out}"
}

CLIENT_DATA_PATHS="$(build_client_data_paths "${HETERO_DATA_ROOT}")"

# Round 1 has no previous personalized adapter. If TRAINING_ITERS > 1, later
# shell-level iterations reuse adapters produced by build_adapters.py.
CLIENT_ADAPTER_PATHS=""

for ((iter=0; iter<TRAINING_ITERS; iter++)); do
  echo "================ Shell-level Iteration ${iter} ================"

  ITER_SAVE_PATH="${SAVE_ROOT}/iter_${iter}_output"
  mkdir -p "${ITER_SAVE_PATH}"

  EXTRA_ADAPTER_ARGS=()
  if [[ -n "${CLIENT_ADAPTER_PATHS}" ]]; then
    EXTRA_ADAPTER_ARGS+=(--client_init_adapter_paths "${CLIENT_ADAPTER_PATHS}")
    EXTRA_ADAPTER_ARGS+=(--client_ref_adapter_paths "${CLIENT_ADAPTER_PATHS}")
  fi

  REF_LOAD_IN_4BIT_ARGS=()
  if [[ "${REF_LOAD_IN_4BIT}" = "1" ]]; then
    REF_LOAD_IN_4BIT_ARGS+=(--ref_load_in_4bit)
  fi

  deepspeed --include "${DEEPSPEED_INCLUDE}" \
    --master_port $((MASTER_PORT_BASE + iter)) \
    "${MAIN_PY}" \
    --pretrain "${POLICY_MODEL_PATH}" \
    --rounds 1 \
    --num_clients "${NUM_CLIENTS}" \
    --client_ids "${CLIENT_IDS}" \
    --dataset_prefix "${HETERO_DATA_ROOT}" \
    --client_data_paths "${CLIENT_DATA_PATHS}" \
    --data_iter "${DATA_ITER}" \
    --save_path "${ITER_SAVE_PATH}" \
    --save_steps -1 \
    --logging_steps 1 \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --prompt_key query \
    --chosen_key chosen_response \
    --rejected_key reject_response \
    --max_len "${MAX_LEN}" \
    --max_samples "${MAX_SAMPLES}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_warmup_ratio 0.01 \
    --zero_stage 2 \
    --beta "${BETA}" \
    --nll_loss_coef "${NLL_LOSS_COEF}" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --gradient_checkpointing \
    --bf16 \
    --ref_offload \
    "${REF_LOAD_IN_4BIT_ARGS[@]}" \
    --attn_implementation flash_attention_2 \
    --use_wandb True \
    --fed_alg "${FED_ALG}" \
    --fedavg_uniform \
    "${EXTRA_ADAPTER_ARGS[@]}"

  # Each DeepSpeed run uses --rounds 1, so the output is fixed here.
  ROUND_DIR="${ITER_SAVE_PATH}/round_1"
  NEXT_ADAPTER_ROOT="${SAVE_ROOT}/iter_${iter}_next_adapters"

  python "${OFFLINE_ITER_PY}" \
    --round_dir "${ROUND_DIR}" \
    --client_ids "${CLIENT_IDS}" \
    --output_dir "${NEXT_ADAPTER_ROOT}"

  CLIENT_ADAPTER_PATHS="$(cat "${NEXT_ADAPTER_ROOT}/client_adapter_paths.txt")"
  echo "[Next Iter Adapter Paths] ${CLIENT_ADAPTER_PATHS}"
done

echo "Shell-level iterative heterogeneous Federated DPO Training Completed!"
