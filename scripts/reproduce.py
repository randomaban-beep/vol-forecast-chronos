"""One-command report rebuild; expensive full experiments are explicit and isolated."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import time
import pandas as pd


def main() -> None:
    """Rebuild the report by default, optionally rescore caches or rerun experiments."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rescore',action='store_true')
    parser.add_argument('--experiments',action='store_true')
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    if args.experiments:
        destination=project/'data'/'reproduction'
        if destination.exists():
            raise SystemExit('data/reproduction already exists. Preserve or rename it before a fresh experiment reproduction.')
        destination.mkdir(parents=True)
        for folder in ['src','scripts','configs','tests','results','docs']:
            shutil.copytree(project/folder,destination/folder,ignore=shutil.ignore_patterns('__pycache__'))
        for file in ['README.md','pyproject.toml','AGENTS.md','LICENSE','CITATION.cff']:
            shutil.copy2(project/file,destination/file)
        for script in ['download_data.py','run_baselines.py','run_zero_shot.py','run_finetune.py','reproduce_stage5.py','package_results.py','compute_extended_metrics.py','make_readme_figures.py','write_readme.py']:
            subprocess.run([sys.executable,str(destination/'scripts'/script)],cwd=destination,check=True)
        print(f'Fresh experiment outputs: {destination}')
        return
    if args.rescore:
        subprocess.run([sys.executable,'scripts/package_results.py'],check=True,cwd=project)
        subprocess.run([sys.executable,'scripts/compute_extended_metrics.py'],check=True,cwd=project)
    start=time.perf_counter()
    for script in ['make_readme_figures.py','write_readme.py']:
        subprocess.run([sys.executable,f'scripts/{script}'],check=True,cwd=project)
    elapsed=time.perf_counter()-start
    if args.rescore:
        runtime=pd.read_csv(project/'results'/'stage6_runtime.csv')
        runtime.loc[runtime.stage.eq(6),'seconds']=elapsed
        runtime.loc[runtime.stage.eq(6),'component']='Measured CSV-only figure + README build; excludes rescore'
        runtime.to_csv(project/'results'/'stage6_runtime.csv',index=False)
        subprocess.run([sys.executable,'scripts/write_readme.py'],check=True,cwd=project)
    print(f'CSV-only report build: {elapsed:.2f} seconds. No models trained.')


if __name__=='__main__':
    main()
