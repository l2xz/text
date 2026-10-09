#!/usr/bin/env bash
set -euo pipefail

# Launch the three-round FEDCARE training pipeline.
# Example:
#   PROJECT_ROOT="$(pwd)" POLICY_MODEL_PATH=/path/to/base_model \
#   GPU_IDS="0 1 2 3" DEEPSPEED_INCLUDE="localhost:0,1,2,3" \
#   bash train_math/run.sh
# Resume from a later round with:
#   START_STAGE=1 END_STAGE=2 bash train_math/run.sh

PROJECT_ROOT=${PROJECT_ROOT:-/path/to/FEDCARE}
SCRIPT_ROOT="${PROJECT_ROOT}/train_math"
source "${SCRIPT_ROOT}/common.sh"

VARIANT=${1:-ours}
if [[ "${VARIANT}" != "ours" ]]; then
  echo "[ERROR] This release contains only FEDCARE (ours)." >&2
  exit 1
fi

START_STAGE=${START_STAGE:-0}
END_STAGE=${END_STAGE:-2}
BUILD_DATA=${BUILD_DATA:-1}
if (( START_STAGE < 0 || END_STAGE > 2 || START_STAGE > END_STAGE )); then
  echo "[ERROR] Invalid stage range: ${START_STAGE} to ${END_STAGE}." >&2
  exit 1
fi

# All generated artifacts are kept under a short, self-contained run directory.
RUN_ROOT=${RUN_ROOT:-${PROJECT_ROOT}/runs/${VARIANT}}
SAVE_ROOT=${SAVE_ROOT:-${RUN_ROOT}/adapters}
DATA_ROOT=${DATA_ROOT:-${RUN_ROOT}/data}
SCORED_ROOT=${SCORED_ROOT:-${RUN_ROOT}/scores}
ROUND1_DATA_ROOT=${ROUND1_DATA_ROOT:-${DATA_ROOT}/r1}
ROUND2_DATA_ROOT=${ROUND2_DATA_ROOT:-${DATA_ROOT}/r2}
ROUND3_DATA_ROOT=${ROUND3_DATA_ROOT:-${DATA_ROOT}/r3}
ROUND1_SCORED_ROOT=${ROUND1_SCORED_ROOT:-${SCORED_ROOT}/r1}
ROUND2_SCORED_ROOT=${ROUND2_SCORED_ROOT:-${SCORED_ROOT}/r2}
ROUND3_SCORED_ROOT=${ROUND3_SCORED_ROOT:-${SCORED_ROOT}/r3}

export PROJECT_ROOT SAVE_ROOT
export ROUND1_DATA_ROOT ROUND2_DATA_ROOT ROUND3_DATA_ROOT
export ROUND1_SCORED_ROOT ROUND2_SCORED_ROOT ROUND3_SCORED_ROOT
export POLICY_MODEL_PATH="$(resolve_model_path "${POLICY_MODEL_PATH:-${BASE_MODEL_PATH:-}}")"
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-${POLICY_MODEL_PATH}}
export GPU_IDS=${GPU_IDS:-"0 1 2 3"}
export DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-localhost:0,1,2,3}
export CLIENT_IDS=${CLIENT_IDS:-0,1,2,3,4,5,6}
export NUM_CLIENTS=${NUM_CLIENTS:-7}
export CLIENT_IDS_LIST=${CLIENT_IDS//,/ }

# Shared training and curriculum defaults. Override any value at launch time.
export FED_ALG=${FED_ALG:-fedavg}
export LEARNING_RATE=${LEARNING_RATE:-2e-6}
export NLL_LOSS_COEF=${NLL_LOSS_COEF:-0.0}
export REFERENCE_MODE=${REFERENCE_MODE:-previous_adapter}
export CURRICULUM_SCORE_MODE=${CURRICULUM_SCORE_MODE:-bees}
export TARGET_SIZE=${TARGET_SIZE:-1000}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-1}
export REF_LOAD_IN_4BIT=${REF_LOAD_IN_4BIT:-0}

mkdir -p "${SAVE_ROOT}" "${DATA_ROOT}" "${SCORED_ROOT}"

configure_builder() {
  local stage="$1"

  unset POLICY_ADAPTER_PATH REF_ADAPTER_PATH REF_ADAPTER_ROOT
  export USE_TRAIN_STATS=1
  export REQUIRE_TRAIN_STATS=1

  case "${stage}" in
    1)
      export POLICY_ADAPTER_ROOT="${SAVE_ROOT}/iter_0_next_adapters"
      export TRAIN_STATS_ROOT="${SAVE_ROOT}/iter_0_output/round_1/client_lora_states"
      ;;
    2)
      export POLICY_ADAPTER_ROOT="${SAVE_ROOT}/iter_1_next_adapters"
      export REF_ADAPTER_ROOT="${SAVE_ROOT}/iter_0_next_adapters"
      export TRAIN_STATS_ROOT="${SAVE_ROOT}/iter_1_output/round_1/client_lora_states"
      ;;
  esac
}

build_stage() {
  local stage="$1"
  if [[ "${BUILD_DATA}" != "1" ]]; then
    echo "[Skip data build] BUILD_DATA=${BUILD_DATA}"
    return
  fi

  case "${stage}" in
    0) bash "${SCRIPT_ROOT}/build_r1.sh" ;;
    1) configure_builder 1; bash "${SCRIPT_ROOT}/build_r2.sh" ;;
    2) configure_builder 2; bash "${SCRIPT_ROOT}/build_r3.sh" ;;
  esac
}

train_stage() {
  local stage="$1"
  case "${stage}" in
    0) TRAINING_ITERS=1 DATA_ITER=0 bash "${SCRIPT_ROOT}/train_r1.sh" ;;
    1) DATA_ITER=1 bash "${SCRIPT_ROOT}/train_r2.sh" ;;
    2) DATA_ITER=2 bash "${SCRIPT_ROOT}/train_r3.sh" ;;
  esac
}

for ((stage=START_STAGE; stage<=END_STAGE; stage++)); do
  echo "================ Round $((stage + 1)): data construction ================"
  build_stage "${stage}"
  echo "================ Round $((stage + 1)): FEDCARE training ================"
  train_stage "${stage}"
done

echo "Training completed. Adapters: ${SAVE_ROOT}"
