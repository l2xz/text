# CodeXGLUE Code-to-Text Data

This directory prepares the six CodeXGLUE code-to-text language clients:
`go`, `java`, `javascript`, `php`, `python`, and `ruby`.

| Script | Purpose |
|---|---|
| `sample.py` | Generate candidate docstrings with a local model. |
| `build.py` | Form DPO pairs from gold and candidate docstrings. |
| `split.py` | Create easy, medium, and hard client curriculum pools. |

The scripts load `google/code_x_glue_ct_code_to_text` directly, or use a local
dataset directory passed through `--local_dataset_root`.

## 1. Sample Candidate Docstrings

```bash
python generate/code_text/sample.py \
  --model_path /path/to/model \
  --output_root data/code_text/candidates \
  --split train \
  --languages all \
  --num_samples 12
```

## 2. Build DPO Pairs

```bash
python generate/code_text/build.py \
  --output_root data/code_text/dpo \
  --generated_candidates data/code_text/candidates/generated_candidates_part0.jsonl \
  --split train \
  --languages all \
  --negative_source model
```

## 3. Create Curriculum Pools

```bash
python generate/code_text/split.py \
  --input_root data/code_text/dpo \
  --private_root client/curriculum \
  --public_root client/supply \
  --public_mode empty \
  --split_strategy balanced_quantile
```

The resulting `client/curriculum/` and `client/supply/` directories are used by
`train_math/run.sh`.
