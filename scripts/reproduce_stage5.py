"""One-command Stage 5 reproduction; prerequisites are frozen Stage 4 caches."""
import subprocess
import sys


def main() -> None:
    """Resume training, score, and regenerate all Stage 5 analysis artifacts."""
    for script in ['run_ablations.py', 'summarize_ablations.py', 'plot_ablations.py', 'document_stage5.py']:
        subprocess.run([sys.executable, f'scripts/{script}'], check=True)


if __name__ == '__main__':
    main()
