# FEDCARE Supplementary Code

This package contains the source code for FEDCARE, a federated curriculum
framework for preference optimization of reasoning-oriented large language
models.

## Included

- Federated DPO training and Dual-LoRA aggregation implementations.
- Curriculum construction, margin scoring, and BeeS-style data-selection tools.
- Data construction for MATH client pools and CodeXGLUE code-to-text clients.
- A small patch that records local DPO training statistics for curriculum selection.

## Excluded

The release intentionally excludes all datasets, pretrained models, LoRA
checkpoints, training outputs, cached files, logs, evaluation results, and
third-party framework or evaluation-harness source trees. Users must obtain
benchmark datasets, model checkpoints, OpenRLHF, and evaluation harnesses
separately in accordance with their respective licenses.

## Installation

```bash
conda create -n fedcare python=3.10 -y
conda activate fedcare
pip install -e .
pip install -r requirements-release.txt
```

Install OpenRLHF and apply the small FEDCARE training-statistics patch
before launching training. See [THIRD_PARTY.md](THIRD_PARTY.md).

Install a PyTorch build matching the local CUDA version before installing the
remaining requirements. FlashAttention and vLLM should be installed following
their official CUDA-compatible installation instructions.

## Configuration

All paths must be supplied by the user. Before launching an experiment, set at
least the following variables:

```bash
export PROJECT_ROOT="$(pwd)"
export POLICY_MODEL_PATH="/path/to/base_model"
export BASE_MODEL_PATH="${POLICY_MODEL_PATH}"
export REF_MODEL_PATH="${POLICY_MODEL_PATH}"
```

The release includes [data/README.md](data/README.md), which documents the DPO
record format and the required MATH and CodeXGLUE client-pool layouts. The
`data/` directory is otherwise ignored by Git, so downloaded datasets are not
part of this release.

## Example

The three-round FEDCARE workflow is launched by:

```bash
PROJECT_ROOT="$(pwd)" \
POLICY_MODEL_PATH="/path/to/base_model" \
GPU_IDS="0 1 2 3" \
DEEPSPEED_INCLUDE="localhost:0,1,2,3" \
bash train_math/run.sh ours
```

The launcher constructs the curriculum data and trains all three rounds. By
default, generated artifacts are written to `runs/ours/`:

```text
runs/ours/adapters/  # Per-round personalized LoRA adapters
runs/ours/data/      # Constructed curriculum data
runs/ours/scores/    # Margin-scoring outputs
```

To resume from an existing first-round adapter, run:

```bash
START_STAGE=1 END_STAGE=2 BUILD_DATA=0 bash train_math/run.sh
```

`BUILD_DATA=0` reuses existing curriculum files. The scripts in `generate/`
prepare the initial private and public preference pools under `client/`.

## Privacy Note

This package contains no local datasets, experiment outputs, model weights, or
user-specific filesystem paths. Placeholder paths such as `/path/to/base_model`
must be replaced with paths on the user's own machine.
