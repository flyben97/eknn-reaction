# eKNN Reaction Prediction

eKNN trains a graph neural network on reactions represented by one or more SMILES strings. At prediction time it finds nearby training reactions in embedding space and adjusts the base prediction using their temperature-weighted residuals. This repository contains the v0.3.0 implementation and research workflows for HiTEA, photoredox, PhotoCatDB and Michael reactions. [中文说明](README_zh.md)

## Run the CPU example

From the repository root, using Python 3.10 on Linux or macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-tested.txt -e .
python examples/quickstart.py
```

The example generates 40 artificial reactions, uses 32 for training and eight for testing, and trains a small GCN. Under `results/synthetic_demo/model/` you will find per-reaction predictions (CSV) and metrics (JSON). The targets come from a formula and only exercise the software. To repeat the run, pass a fresh directory with `--out-dir`.

## Train on measured reactions

Prepare separate training and held-out CSVs with the same SMILES columns and a numeric target. For example, given columns named `reactant_smiles`, `catalyst_smiles` and `Yield`, run:

```bash
eknn-train --dataset my_reactions \
  --train-csv train.csv --test-csv test.csv \
  --smiles-cols reactant_smiles catalyst_smiles --target-col Yield \
  --backbone gcn --no-cache-graphs --out-dir results/my_reactions \
  --set train.device=cpu train.seed=42 data.seed=42
```

The runner selects neighbour settings on internal validation data and refits the base model on the full training set. Its final memory uses training-set residuals. For group-based internal validation, add `--group-col COLUMN --group-val` and define the outer split separately. Use `eknn-train --help` for all options. Registered datasets can be located with `EKNN_DATA_DIR`. [Data requirements](docs/DATA.md)

## Research workflows and tests

The model and retrieval library is in [eknn_reaction/](eknn_reaction/). Task-specific [scripts](scripts/) implement HiTEA screen-level calibration, photoredox substrate and paper splits, PhotoCatDB prediction, and Aho/Michael data preparation. They use external inputs. The CPU example and unit tests generate their inputs locally.

```bash
python -m pip install -c requirements-tested.txt -e '.[research,dev]'
python -m pytest -q
```

Released under the [MIT License](LICENSE). Attribution details are in [NOTICE](NOTICE).
