"""Read the bundled numerical summaries without training or binary artifacts."""
from pathlib import Path
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'data/results/npi_evaluation_controls_20260929/matched_comparisons.csv'
    rows = list(csv.DictReader(source.open(encoding='utf-8')))
    summary = {}
    for key, label in [('cpu_length', 'single_cpu'), ('hard_length', 'full_correction_cpu'),
                       ('split_length', 'relaxed_correction_cpu'), ('gpu_length', 'single_gpu')]:
        selected = [r for r in rows if r['comparison'] == key and r['time_steps'] == '960']
        seeds = sorted(int(r['seed']) for r in selected)
        if seeds != list(range(701, 707)):
            raise ValueError(f'Incomplete or duplicated six-seed comparison: {key}')
        gains = [float(r['gain_bp']) for r in selected]
        summary[label] = dict(seeds=seeds, median_ce_change_bp=statistics.median(gains),
            adjusted_decreases=sum(float(r['adjusted_high_bp']) < 0 for r in selected),
            adjusted_increases=sum(float(r['adjusted_low_bp']) > 0 for r in selected))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
