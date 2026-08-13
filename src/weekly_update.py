from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(name, command):
    """Run one pipeline step and stop if it fails."""

    print("=" * 60)
    print(name)
    print("=" * 60)

    result = subprocess.run([sys.executable, "-m", *command], cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def main():
    print("=" * 60)
    print("Fantasy Pitcher Predictor — Weekly Update")
    print("=" * 60)

    run_step("STEP 1 — Retraining 2026 model", ["src.retraining"])

    run_step("STEP 2 — Generating daily predictions", ["src.daily_update"])

    print("=" * 60)
    print("Weekly update complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()