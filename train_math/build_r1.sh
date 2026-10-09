#!/bin/bash
set -e
set -x

PROJECT_ROOT=${PROJECT_ROOT:-/path/to/FEDCARE}
source "${PROJECT_ROOT}/train_math/common.sh"
SCORER="${PROJECT_ROOT}/utils/curriculum/score.py"
ALLOCATOR="${PROJECT_ROOT}/utils/curriculum/select.py"

LOCAL_ROOT=${LOCAL_ROOT:-${PROJECT_ROOT}/client/curriculum}
PUBLIC_ROOT=${PUBLIC_ROOT:-${PROJECT_ROOT}/client/supply}
OUTPUT_ROOT=${ROUND1_DATA_ROOT:-${PROJECT_ROOT}/runs/ours/data/r1}
SCORED_ROOT=${ROUND1_SCORED_ROOT:-${PROJECT_ROOT}/runs/ours/scores/r1}
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/path/to/base_model}"
POLICY_MODEL_PATH="$(resolve_model_path "${POLICY_MODEL_PATH}")"

# Round 1: model-selected easy warm-up.
# There is no client LoRA before round 1, so scoring uses the base policy
# chosen-vs-rejected logratio (--ref_mode none). The allocator keeps local easy
# data first and only uses public easy as a capped supplement.
TARGET_SIZE=${TARGET_SIZE:-1000}
MAX_PUBLIC_COUNT=${MAX_PUBLIC_COUNT:--1}
MAX_PUBLIC_RATIO=${MAX_PUBLIC_RATIO:-0.50}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-8}
SCORE_GPU_MEMORY=${SCORE_GPU_MEMORY:-0.65}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
SCORE_ENFORCE_EAGER=${SCORE_ENFORCE_EAGER:-0}
SCORE_OOM_RETRY=${SCORE_OOM_RETRY:-1}
RETRY_SCORE_BATCH_SIZE=${RETRY_SCORE_BATCH_SIZE:-1}
RETRY_SCORE_GPU_MEMORY=${RETRY_SCORE_GPU_MEMORY:-0.35}
RETRY_MAX_NUM_SEQS=${RETRY_MAX_NUM_SEQS:-1}
RETRY_MAX_NUM_BATCHED_TOKENS=${RETRY_MAX_NUM_BATCHED_TOKENS:-1024}
FINAL_RETRY_SCORE_BATCH_SIZE=${FINAL_RETRY_SCORE_BATCH_SIZE:-1}
FINAL_RETRY_SCORE_GPU_MEMORY=${FINAL_RETRY_SCORE_GPU_MEMORY:-0.30}
FINAL_RETRY_MAX_NUM_SEQS=${FINAL_RETRY_MAX_NUM_SEQS:-1}
FINAL_RETRY_MAX_NUM_BATCHED_TOKENS=${FINAL_RETRY_MAX_NUM_BATCHED_TOKENS:-1024}
RETRY_SLEEP_SECONDS=${RETRY_SLEEP_SECONDS:-20}
BATCH_RELEASE_SLEEP_SECONDS=${BATCH_RELEASE_SLEEP_SECONDS:-20}
SCORE_MAX_ROWS_PER_SOURCE=${SCORE_MAX_ROWS_PER_SOURCE:--1}
SCORE_ROW_SAMPLE_MODE=${SCORE_ROW_SAMPLE_MODE:-first}
PUBLIC_THRESHOLD_QUANTILE=${PUBLIC_THRESHOLD_QUANTILE:-0.25}
CURRICULUM_SCORE_MODE=${CURRICULUM_SCORE_MODE:-bees}
CURRICULUM_RISK_LAMBDA=${CURRICULUM_RISK_LAMBDA:-1.0}
BEES_MARGIN_KEYS=${BEES_MARGIN_KEYS:-client_margin,reward_gap,implicit_reward_gap,teacher_margin,dpo_margin,margin,margins}
BEES_MARGIN_LOW=${BEES_MARGIN_LOW:--0.5}
BEES_MARGIN_HIGH=${BEES_MARGIN_HIGH:-0.5}
BEES_MIN_PROB=${BEES_MIN_PROB:-0.55}
BEES_PROB_FLOOR=${BEES_PROB_FLOOR:-0.0001}
BEES_MIN_SOURCES=${BEES_MIN_SOURCES:-1}
RESCORE=${RESCORE:-1}
CLIENT_IDS_RAW=${CLIENT_IDS:-"0 1 2 3 4 5 6"}
GPU_IDS_RAW=${GPU_IDS:-"4 5 6 7"}
CLIENT_IDS=(${CLIENT_IDS_RAW//,/ })
GPU_IDS=(${GPU_IDS_RAW//,/ })

if [ "${#GPU_IDS[@]}" -le 0 ]; then
  echo "ERROR: GPU_IDS cannot be empty."
  exit 1
fi

if [ ! -f "${SCORER}" ]; then
  echo "ERROR: missing scorer script: ${SCORER}"
  exit 1
fi

if [ ! -f "${ALLOCATOR}" ]; then
  echo "ERROR: missing allocator script: ${ALLOCATOR}"
  exit 1
fi

for cid in "${CLIENT_IDS[@]}"
do
  if [ ! -f "${LOCAL_ROOT}/client_${cid}/curri_easy.jsonl" ]; then
    echo "ERROR: missing local easy data: ${LOCAL_ROOT}/client_${cid}/curri_easy.jsonl"
    exit 1
  fi
  if [ ! -f "${PUBLIC_ROOT}/client_${cid}/curri_easy.jsonl" ]; then
    echo "ERROR: missing public easy data: ${PUBLIC_ROOT}/client_${cid}/curri_easy.jsonl"
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${SCORED_ROOT}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

run_scorer() {
  local gpu_id="$1"
  local batch_size="$2"
  local gpu_memory="$3"
  local max_num_seqs="$4"
  local max_num_batched_tokens="$5"
  local enforce_eager="$6"
  shift 6

  local eager_args=()
  if [ "${enforce_eager}" = "1" ]; then
    eager_args+=(--enforce_eager)
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" python "${SCORER}" \
    "$@" \
    --batch_size "${batch_size}" \
    --gpu_memory_utilization "${gpu_memory}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --max_num_seqs "${max_num_seqs}" \
    --max_num_batched_tokens "${max_num_batched_tokens}" \
    "${eager_args[@]}"
}

run_one_client() {
  local cid="$1"
  local gpu_id="$2"
  local client_scored_dir="${SCORED_ROOT}/client_${cid}"

  echo "========== client_${cid} uses physical GPU ${gpu_id} =========="
  mkdir -p "${client_scored_dir}"
  mkdir -p "${OUTPUT_ROOT}/client_${cid}"

  if [ "${RESCORE}" = "1" ] || [ ! -f "${client_scored_dir}/local_easy.jsonl" ] || [ ! -f "${client_scored_dir}/public_easy.jsonl" ]; then
    SCORE_ARGS=(
      --input "local_easy=${LOCAL_ROOT}/client_${cid}/curri_easy.jsonl" \
      --input "public_easy=${PUBLIC_ROOT}/client_${cid}/curri_easy.jsonl" \
      --output_dir "${client_scored_dir}" \
      --model "${POLICY_MODEL_PATH}" \
      --ref_mode none \
      --beta 0.1 \
      --max_model_len 1024 \
      --max_rows_per_source "${SCORE_MAX_ROWS_PER_SOURCE}" \
      --row_sample_mode "${SCORE_ROW_SAMPLE_MODE}" \
      --row_sample_seed $((20260603 + cid))
    )

    if ! run_scorer "${gpu_id}" "${SCORE_BATCH_SIZE}" "${SCORE_GPU_MEMORY}" "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}" "${SCORE_ENFORCE_EAGER}" "${SCORE_ARGS[@]}"; then
      if [ "${SCORE_OOM_RETRY}" != "1" ]; then
        return 1
      fi
      echo "[Retry client_${cid}] scorer failed; retrying with low-memory settings."
      sleep "${RETRY_SLEEP_SECONDS}"
      if ! run_scorer "${gpu_id}" "${RETRY_SCORE_BATCH_SIZE}" "${RETRY_SCORE_GPU_MEMORY}" "${RETRY_MAX_NUM_SEQS}" "${RETRY_MAX_NUM_BATCHED_TOKENS}" 1 "${SCORE_ARGS[@]}"; then
        echo "[Final retry client_${cid}] scorer failed again; retrying with minimal-memory settings."
        sleep "${RETRY_SLEEP_SECONDS}"
        run_scorer "${gpu_id}" "${FINAL_RETRY_SCORE_BATCH_SIZE}" "${FINAL_RETRY_SCORE_GPU_MEMORY}" "${FINAL_RETRY_MAX_NUM_SEQS}" "${FINAL_RETRY_MAX_NUM_BATCHED_TOKENS}" 1 "${SCORE_ARGS[@]}"
      fi
    fi
  fi

  python "${ALLOCATOR}" \
    --source "local_easy=${client_scored_dir}/local_easy.jsonl" \
    --source "public_easy=${client_scored_dir}/public_easy.jsonl" \
    --output "${OUTPUT_ROOT}/client_${cid}/dpo_train.jsonl" \
    --target_size "${TARGET_SIZE}" \
    --score_mode "${CURRICULUM_SCORE_MODE}" \
    --risk_lambda "${CURRICULUM_RISK_LAMBDA}" \
    --bees_margin_keys "${BEES_MARGIN_KEYS}" \
    --bees_margin_low "${BEES_MARGIN_LOW}" \
    --bees_margin_high "${BEES_MARGIN_HIGH}" \
    --bees_min_prob "${BEES_MIN_PROB}" \
    --bees_prob_floor "${BEES_PROB_FLOOR}" \
    --bees_min_sources "${BEES_MIN_SOURCES}" \
    --public_threshold_quantile "${PUBLIC_THRESHOLD_QUANTILE}" \
    --max_public_count "${MAX_PUBLIC_COUNT}" \
    --max_public_ratio "${MAX_PUBLIC_RATIO}" \
    --drop_truncated \
    --dedup \
    --seed $((20260520 + cid))

  echo "========== client_${cid} finished on GPU ${gpu_id} =========="
}

trap 'echo "Interrupted. Killing background jobs..."; jobs -p | xargs -r kill; exit 1' INT TERM

pids=()
failed=0

wait_batch() {
  local batch_failed=0
  local pid
  for pid in "$@"
  do
    if ! wait "${pid}"; then
      echo "ERROR: one client process failed. pid=${pid}"
      batch_failed=1
    fi
  done
  return "${batch_failed}"
}

for idx in "${!CLIENT_IDS[@]}"
do
  cid="${CLIENT_IDS[$idx]}"
  gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"

  run_one_client "${cid}" "${gpu_id}" > "${SCORED_ROOT}/client_${cid}.log" 2>&1 &
  pids+=("$!")

  if [ "${#pids[@]}" -ge "${#GPU_IDS[@]}" ]; then
    if ! wait_batch "${pids[@]}"; then
      failed=1
    fi
    pids=()
    sleep "${BATCH_RELEASE_SLEEP_SECONDS}"
  fi
done

if [ "${#pids[@]}" -gt 0 ]; then
  if ! wait_batch "${pids[@]}"; then
    failed=1
  fi
fi

if [ "${failed}" -ne 0 ]; then
  echo "ERROR: at least one client failed. Check logs in ${SCORED_ROOT}/client_*.log"
  for log_file in "${SCORED_ROOT}"/client_*.log
  do
    if [ -f "${log_file}" ]; then
      echo "================ tail ${log_file} ================"
      tail -n 80 "${log_file}" || true
    fi
  done
  exit 1
fi

echo "Round-1 model-selected easy curriculum data written to: ${OUTPUT_ROOT}"
