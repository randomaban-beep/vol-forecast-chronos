"""Final report numbers, causal exclusions and local image integrity."""
from pathlib import Path
import re
import subprocess
import hashlib
import numpy as np
import pandas as pd
from src.report import headline_table, headline_rows
from scripts.package_results import excluded_windows


def test_readme_headline_exactly_matches_csvs():
    text=Path('README.md').read_text(encoding='utf-8')
    block=text.split('<!-- headline-table-start -->')[1].split('<!-- headline-table-end -->')[0].strip()
    assert block==headline_table()
    # Independent source lookup guards accidental renderer-only consistency.
    original=pd.read_csv('results/stage2_baselines.csv')
    values=original.loc[original.model.eq('HAR-RV')].set_index(['symbol_group','horizon'])
    for line in block.splitlines()[2:]:
        fields=[part.strip() for part in line.strip('|').split('|')]
        assert fields[3]==f'{values.loc[(fields[0],int(fields[1])), "qlike_raw"]:.4f}'


def test_all_markdown_images_are_local_and_exist():
    text=Path('README.md').read_text(encoding='utf-8')
    images=re.findall(r'!\[([^]]*)\]\(([^)]+)\)',text)
    assert images
    for alt,path in images:
        assert alt.strip()
        assert not re.match(r'\w+://|/|[A-Za-z]:',path)
        assert Path(path).is_file(),path


def test_robustness_excludes_overlapping_targets_not_just_end_dates():
    rows=pd.DataFrame({'origin':pd.to_datetime(['2025-10-09','2025-10-12','2025-09-19','2025-10-08'],utc=True),
        'target_end':pd.to_datetime(['2025-10-10','2025-10-13','2025-10-11','2025-10-09'],utc=True)})
    assert excluded_windows(rows).tolist()==[True,False,True,False]


def test_robustness_full_sample_matches_original_summary():
    baseline=headline_rows().set_index(['model','symbol_group','horizon'])
    robust=pd.read_csv('results/stage6_robustness.csv')
    for row in robust.loc[robust['sample'].eq('all dates')].itertuples():
        original=baseline.loc[(row.model,row.symbol_group,row.horizon)]
        np.testing.assert_allclose([row.rmse,row.mae,row.qlike_raw,row.qlike],
            original[['rmse','mae','qlike_raw','qlike']].to_numpy(dtype=float),rtol=1e-12,atol=1e-12)


def test_historical_results_preserved():
    audit=pd.read_csv('results/stage6_preservation.csv')
    assert len(audit)>0
    for row in audit.itertuples():
        assert row.before_sha256==row.after_sha256
        assert hashlib.sha256(Path(row.path).read_bytes()).hexdigest()==row.before_sha256


def test_extended_metrics_sanity():
    path = Path('results/stage6_extended_metrics.csv')
    assert path.is_file()
    df = pd.read_csv(path)
    assert len(df) == 48
    assert set(df.symbol_group.unique()) == {'fit-eligible', 'held-out'}
    assert set(df.horizon.unique()) == {1, 5, 22}
    assert (df.mda_pct >= 0).all() and (df.mda_pct <= 100).all()
    assert (df.mdape_pct > 0).all()
    assert (df.r2_mz_pct >= 0).all() and (df.r2_mz_pct <= 100).all()
    assert (df.pinball_90 > 0).all()
