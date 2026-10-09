# Third-Party Dependency Notice

## OpenRLHF

FEDCARE uses the public OpenRLHF
training framework for its model, dataset, distributed-training, and base DPO
trainer implementations. OpenRLHF source code is not included in this release.
OpenRLHF is distributed under the Apache License 2.0; consult its repository
for the applicable license text, source revision, and installation guidance.

FEDCARE adds only one project-specific change to the upstream DPO trainer:
it records averages of local training-only DPO statistics, including the
chosen-rejected margin. The curriculum selector consumes these statistics in
later rounds. This change is provided as
`patches/openrlhf_train_stats.patch`.

## Installation

Install a compatible OpenRLHF checkout, apply the patch from its repository
root, then install the patched checkout in editable mode:

```bash
git clone OpenRLHF /path/to/OpenRLHF
cd /path/to/OpenRLHF
git apply --ignore-space-change /path/to/FEDCARE/patches/openrlhf_train_stats.patch
pip install -e .
```

The patch applies to the upstream `openrlhf/trainer/dpo_trainer.py` interface
used by this release. If the selected OpenRLHF revision has diverged, choose a
compatible revision or port the small, self-contained additions in the patch.

## Math Evaluation

The MATH evaluation harness and its LaTeX parser are third-party components and
are not included in this anonymous release. Obtain a compatible evaluation
harness separately when reproducing the reported mathematical-reasoning
metrics.
