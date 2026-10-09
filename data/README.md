# Data Guide

No datasets are distributed with this release. Download or prepare the source
data separately, then create the client curriculum pools expected by the
training launcher.

## Required DPO Format

Each training file is JSONL. Every record must provide these fields:

```json
{
  "query": "instruction or problem",
  "chosen_response": "preferred response",
  "reject_response": "dispreferred response"
}
```

Additional metadata fields are optional.

## MATH Client Pools

Before running `train_math/run.sh`, provide private and public candidate pools
outside this `data/` directory, using the locations below by default:

```text
client/
  curriculum/
    client_0/
      curri_easy.jsonl
      curri_medium.jsonl
      curri_hard_gap_selected.jsonl
    client_1/
      ...
  supply/
    client_0/
      curri_easy.jsonl
      curri_medium.jsonl
    client_1/
      ...
```

The number of client directories must match `CLIENT_IDS` and `NUM_CLIENTS`.
Set `LOCAL_ROOT` and `PUBLIC_ROOT` to use alternative locations.

## CodeXGLUE Code-to-Text

The code-to-text preparation scripts are in `generate/code_text/`:

1. Run `sample.py` to generate candidate docstrings.
2. Run `build.py` to create one `dpo_train.jsonl` file per language.
3. Run `split.py` to create the `client/curriculum/` and `client/supply/`
   pools consumed by `train_math/run.sh`.

See `generate/code_text/README.md` for the commands and arguments. Generated
datasets, model weights, checkpoints, and experiment outputs must not be added
to this release.
