# Data inputs

The graph runner reads separate training and held-out CSV files. Both files must contain the SMILES columns given to `--smiles-cols` and the numeric target given to `--target-col`. For grouped validation, provide the grouping column with `--group-col` and enable `--group-val`. Assign outer train and test groups before model selection.

[The dataset registry](../eknn_reaction/data/datasets.py) defines the relative paths, feature columns, targets and grouping keys for named datasets. Set `EKNN_DATA_DIR` to their parent directory. Task-specific scripts may also require a split assignment, a trained checkpoint or calibration data.

The graph runner writes per-reaction predictions and metrics under `--out-dir`. Use the validation-selected neighbour settings when reporting held-out performance.
