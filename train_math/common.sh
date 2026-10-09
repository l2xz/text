#!/bin/bash

model_path_candidates() {
  local requested="${1:-}"
  local candidate

  if [[ -n "${requested}" ]]; then
    echo "${requested}"
  fi

  for candidate in \
    "${BASE_MODEL_PATH:-}" \
    "/path/to/model_cache/modelscope/models/Qwen--Qwen2.5-7B/snapshots/master" \
    "/path/to/base_model"
  do
    if [[ -n "${candidate}" ]]; then
      echo "${candidate}"
    fi
  done

  for candidate in \
    /path/to/model_cache/modelscope/models/Qwen--Qwen2.5-7B/snapshots/* \
    /path/to/model_cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/*
  do
    echo "${candidate}"
  done
}

resolve_model_path() {
  local requested="${1:-}"
  local candidate

  while IFS= read -r candidate; do
    if [[ -n "${candidate}" && -f "${candidate}/config.json" ]]; then
      echo "${candidate}"
      return 0
    fi
  done < <(model_path_candidates "${requested}")

  if [[ -n "${requested}" && "${requested}" != /* ]]; then
    echo "[Warn] No local config.json found for ${requested}; treating it as a HuggingFace repo id." >&2
    echo "${requested}"
    return 0
  fi

  echo "[ERROR] Cannot resolve a valid local model path." >&2
  echo "        Requested: ${requested:-<empty>}" >&2
  echo "        A valid local model directory must contain config.json." >&2
  echo "        Try setting:" >&2
  echo "        POLICY_MODEL_PATH=/path/to/model_cache/modelscope/models/Qwen--Qwen2.5-7B/snapshots/master" >&2
  return 1
}

deepspeed_world_size() {
  local ids
  if [[ -n "${DEEPSPEED_INCLUDE:-}" && "${DEEPSPEED_INCLUDE}" == *":"* ]]; then
    ids="${DEEPSPEED_INCLUDE#*:}"
  else
    ids="${GPU_IDS:-}"
  fi
  ids="${ids//,/ }"
  # shellcheck disable=SC2086
  set -- ${ids}
  echo "$#"
}

ensure_deepspeed_batch_size() {
  local world_size
  local unit
  local adjusted

  world_size="$(deepspeed_world_size)"
  if (( world_size <= 0 )); then
    echo "[ERROR] Cannot infer DeepSpeed world size from DEEPSPEED_INCLUDE='${DEEPSPEED_INCLUDE:-}' or GPU_IDS='${GPU_IDS:-}'." >&2
    return 1
  fi

  unit=$(( MICRO_TRAIN_BATCH_SIZE * world_size ))
  if (( unit <= 0 )); then
    echo "[ERROR] Invalid batch settings: micro=${MICRO_TRAIN_BATCH_SIZE}, world_size=${world_size}" >&2
    return 1
  fi

  if (( TRAIN_BATCH_SIZE % unit != 0 )); then
    adjusted=$(( (TRAIN_BATCH_SIZE / unit) * unit ))
    if (( adjusted <= 0 )); then
      adjusted="${unit}"
    fi
    echo "[Warn] Adjust TRAIN_BATCH_SIZE from ${TRAIN_BATCH_SIZE} to ${adjusted} because DeepSpeed requires:" >&2
    echo "       train_batch_size = micro_train_batch_size * accumulated_gradient * world_size" >&2
    echo "       ${adjusted} = ${MICRO_TRAIN_BATCH_SIZE} * $(( adjusted / unit )) * ${world_size}" >&2
    TRAIN_BATCH_SIZE="${adjusted}"
  fi

  export TRAIN_BATCH_SIZE
}
