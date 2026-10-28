"""Append reproducible Stage 5 tables without editing historical README content."""
from pathlib import Path
import pandas as pd


def markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render compact tables without adding a tabulate dependency."""
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def main() -> None:
    """Build the Stage 5 README section and reusable PR results text."""
    results = Path('results')
    weekday = pd.read_csv(results / 'stage5_weekday.csv')
    extended = pd.read_csv(results / 'stage5_extended_tiny_summary.csv')
    costs = pd.read_csv(results / 'stage5_run_costs.csv')
    extension_cost = costs.loc[costs.extended_budget]
    selection = pd.read_csv(results / 'stage5_frozen_selection.csv')
    preservation = pd.read_csv(results / 'stage5_preservation.csv')
    new = costs.loc[costs.reused_from.isna()]
    text = '''## Stage 5: calendar baseline, ablations and analysis

Reproduce from the frozen Stage 4 caches with `python scripts/reproduce_stage5.py`.
Install plotting support with `pip install -e ".[analysis]"`. Training caches, predictions
and checkpoints remain under ignored `data/`; only new Stage 5 result files are added.
The runner resumes fingerprint-matched completed runs and reuses identical Stage 4 runs.

HAR-weekday adds six weekday dummies (Monday reference) to the existing pooled HAR-RV
on daily/weekly/monthly mean log RV. The h=1 weekday is origin + one day; h=5 and h=22
use the origin weekday. All are calendar-known at the origin. The same six-symbol
expanding fit, 30-day refit dates, targets and common forecast panel are retained.

'''
    text += markdown(weekday, ['symbol_group', 'horizon', 'rmse', 'mae', 'qlike_raw', 'qlike'])
    text += '''

The one-factor small-model grid uses full fine-tuning and LoRA rank 16 for contexts
32/64/128/256 and earliest training-window fractions 10/25/50/100%; the rank sweep
uses 4/8/16/32 with context 256 and all training windows. Three seeds are used everywhere.
Context comparisons retain the same 256-history target-window membership and slice
the trailing context, isolating context length from sample size. Fractions select
floor(fraction × N) windows ordered by origin then symbol; selected samples retain
the original training order. No validation/test/held-out windows enter training.
NaN contexts are retained and masked by Chronos, with no interpolation.

Test curves are reporting only. Before ablation scoring, original Stage 4 families
were compared by mean seed validation RMSE on the six fit-eligible symbols; these
choices are frozen and are not replaced by test winners:

'''
    text += markdown(selection.loc[selection.selected], ['horizon', 'family', 'validation_rmse'])
    text += '''

![Context raw QLIKE](results/stage5_context_qlike_raw.png)
![Training fraction raw QLIKE](results/stage5_fraction_qlike_raw.png)
![Rank raw QLIKE](results/stage5_rank_qlike_raw.png)
![Rank parameter counts](results/stage5_rank_parameters.png)

Calibrated counterparts are saved as `results/stage5_{context,fraction,rank}_qlike.png`.
The full [ablation table](results/stage5_ablation_summary.csv) includes both QLIKE
variants, RMSE/MAE, 95% seed/date block-bootstrap CIs, costs, and HAC/Harvey DM tests
against HAR-RV and HAR-weekday. Metrics average seed-specific losses, not ensemble
predictions. The unchanged shared evaluator applies raw exp(mu) and calibrated
smearing × exp(mu); each smearing factor is fitted only to six-symbol 2024 validation
residuals and frozen. SOL and DOGE never calibrate or choose models. Joint 95% MCS
results include prior model families plus the new experiments; identical reused
loss columns share membership. These multiple comparisons are descriptive and do
not authorize test-based selection.

Regimes use each symbol/horizon's training-only 90th percentile of sqrt(mean RV)
(equivalently a volatility threshold) and classify each test target above it or in
the rest. Held-out histories define only their retrospective regime thresholds.
The [regime table](results/stage5_regimes.csv) compares HAR, HAR-weekday and the frozen
fine-tuned family. Conditional CIs resample whole calendar blocks and divide sampled
loss totals by sampled observation counts. Empty regime dates remain zero-padded
for calendar-aware DM/HAC and MCS; the common count normalization preserves conditional
mean loss differences. Regime MCS uses just these three families. Thresholds and
sample counts are saved separately; future targets never set thresholds.

The three worst h=1 cases are ranked by mean seed squared log error across all symbols.
Plots show mean log forecasts and the range across seeds for illustration; headline
metrics remain means of seed-specific scores.

'''
    text += Path('data/stage5/error_analysis.md').read_text(encoding='utf-8')
    text += '\n\n' + '\n'.join(f'![Worst forecast {i}](results/stage5_worst_{i}.png)' for i in range(1, 4))
    text += '''

The **extended-budget tiny LoRA rank 16** experiment is separate. Both tiny LoRA
families had at least one Stage 4 seed select epoch 3; rank 16 had the lower mean
validation loss (12.1193 vs 12.1517 for rank 8). It restarts from the same pretrained
weights/seeds with the same validation-only early stopping and an eight-epoch cap.
No small-model run is extended, and no Stage 4 number or frozen headline is replaced.

'''
    text += markdown(extension_cost, ['seed', 'best_epoch', 'epochs_completed', 'best_validation_loss'])
    text += '\n\n' + ('At least one extended seed still selected epoch 8.' if extension_cost.best_epoch.eq(8).any()
                        else 'No extended seed selected the final eight-epoch cap.')
    text += '\n\n' + markdown(extended, ['symbol_group', 'horizon', 'rmse', 'mae', 'qlike_raw', 'qlike'])
    text += f'\n\nNew training used {new.fit_seconds.sum()/60:.1f} minutes across {len(new)} runs; '
    text += f'{costs.reused_from.notna().sum()} runs reused Stage 4. Peak training allocation was '
    text += f'{new.gpu_training_peak_mb.max():.0f} MiB on {", ".join(new.device.unique())}. '
    text += 'Per-run costs, best epochs, inference times and raw histories are in the Stage 5 CSVs. '
    text += f'All {len(preservation)} historical result files were verified byte-identical by SHA-256. '
    text += 'Full reproduction from raw data also requires the earlier-stage commands and their runtime.\n'
    readme = Path('README.md')
    original = readme.read_text(encoding='utf-8').split('\n## Stage 5:')[0]
    readme.write_text(original.rstrip()+'\n\n'+text, encoding='utf-8')
    Path('data/stage5/pr_results.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
