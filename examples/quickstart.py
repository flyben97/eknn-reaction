"""Run the full training pipeline on artificial labels, without research data."""
from pathlib import Path
import argparse
import csv
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('results/synthetic_demo'))
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # These labels are an arbitrary formula, not measured yields or energies.
    rows = [{'reactant_smiles': 'C' * n + terminal,
             'catalyst_smiles': '[Na+]', 'synthetic_target': n * 0.07 + j * 0.13}
            for n in range(1, 11) for j, terminal in enumerate(('O', 'N', 'F', 'Cl'))]
    splits = {'train': [r for i, r in enumerate(rows) if i % 5 != 0],
              'test': [r for i, r in enumerate(rows) if i % 5 == 0]}
    for name, records in splits.items():
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(records)
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root/'scripts/train_gnn_eknn.py'),
               '--dataset', 'synthetic_demo', '--train-csv', str(output/'train.csv'),
               '--test-csv', str(output/'test.csv'), '--smiles-cols',
               'reactant_smiles', 'catalyst_smiles', '--target-col', 'synthetic_target',
               '--backbone', 'gcn', '--epochs', '2', '--min-epochs', '1',
               '--warmup-epochs', '0', '--batch-size', '8', '--d-model', '16',
               '--n-layers', '1', '--n-heads', '2', '--ff-dim', '32', '--pe-dim', '0',
               '--no-cache-graphs', '--deterministic', '--out-dir', str(output/'model'),
               '--set', 'train.device=cpu', 'train.seed=42', 'data.seed=42']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run(command, cwd=root, env=env, check=True)
    print(f'Synthetic pipeline completed: {output}')
    print('The labels and metrics are only a software check, not scientific evidence.')


if __name__ == '__main__':
    main()
