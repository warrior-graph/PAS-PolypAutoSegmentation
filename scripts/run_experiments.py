"""Run an experiment matrix (train + several evaluations each) and summarise the results.

    python scripts/run_experiments.py experiments/round2.yaml                 # everything
    python scripts/run_experiments.py experiments/round2.yaml --only e1_b0_strong e2_b0_448
    python scripts/run_experiments.py experiments/round2.yaml --jobs 2        # 2 runs share the GPU
    python scripts/run_experiments.py experiments/round2.yaml --dry-run
    python scripts/run_experiments.py experiments/round2.yaml --bench         # time a short B0 run
    python scripts/run_experiments.py experiments/round2.yaml --summary-only

Finished steps are skipped (a run with test_results.json is not retrained, an existing
eval_<variant>.json is not recomputed), so an interrupted matrix can simply be relaunched.
"""

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
UNSEEN = ("CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300")
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_matrix(path):
    with open(path) as f:
        spec = yaml.safe_load(f)
    defaults = spec.get("defaults", {})
    experiments = {}
    for name, exp in spec["experiments"].items():
        exp = exp or {}
        config = exp.get("config", defaults.get("config"))
        if not config:
            raise SystemExit(f"{path}: experiment {name!r} has no config (set it or defaults.config)")
        experiments[name] = {
            "config": config,
            "set": list(defaults.get("set", [])) + list(exp.get("set", [])),
            "evals": exp.get("evals", defaults.get("evals", {})),
        }
    return experiments


def run(cmd, log_path, echo):
    """Run a command, appending its output to log_path (and stdout when echo); raise on failure."""
    with open(log_path, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        p = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
            if echo:
                print(line, end="", flush=True)
        if p.wait():
            raise RuntimeError(f"exit {p.returncode}: {' '.join(map(str, cmd))} (see {log_path})")


def train_cmd(name, exp, out_dir, data_root):
    return [
        sys.executable, "train.py", "--config", exp["config"], "--set",
        f"experiment={name}", f"output_dir={out_dir}", f"data.root={data_root}", *exp["set"],
    ]


def run_experiment(name, exp, out_dir, data_root, echo, force):
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "runner.log"
    t0 = time.time()
    if force or not (run_dir / "test_results.json").exists():
        log(f"{name}: training ({exp['config']} {' '.join(exp['set'])})")
        run(train_cmd(name, exp, out_dir, data_root), log_path, echo)
    else:
        log(f"{name}: already trained, skipping")
    for variant, flags in exp["evals"].items():
        if force or not (run_dir / f"eval_{variant}.json").exists():
            log(f"{name}: eval {variant} {' '.join(map(str, flags))}")
            cmd = [sys.executable, "eval.py", "--checkpoint", run_dir / "best.pth", "--out", run_dir / f"eval_{variant}", *flags]
            run(cmd, log_path, echo)
    minutes = (time.time() - t0) / 60
    log(f"{name}: done in {minutes:.1f} min")
    return minutes


def collect(out_dir, names):
    """Rows of (experiment, variant, {dataset: mDice}) from finished runs."""
    rows = []
    for name in names:
        run_dir = Path(out_dir) / name
        files = [("clean", run_dir / "test_results.json")] + [
            (p.stem.removeprefix("eval_"), p) for p in sorted(run_dir.glob("eval_*.json"))
        ]
        for variant, path in files:
            if path.exists():
                res = json.loads(path.read_text())["results"]
                rows.append((name, variant, {k: v["mDice"] for k, v in res.items()}))
    return rows


def summarize(out_dir, names):
    rows = collect(out_dir, names)
    if not rows:
        return ""
    datasets = list(rows[0][2])
    unseen = [d for d in datasets if d in UNSEEN]
    header = ["experiment", "eval", *datasets, "mean unseen", "mean all"]
    table = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    with open(Path(out_dir) / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name, variant, m in rows:
            vals = [m.get(d, float("nan")) for d in datasets]
            mu = sum(m[d] for d in unseen) / max(len(unseen), 1)
            ma = sum(vals) / len(vals)
            w.writerow([name, variant, *(f"{v:.4f}" for v in vals), f"{mu:.4f}", f"{ma:.4f}"])
            table.append(f"| {name} | {variant} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {mu:.3f} | {ma:.3f} |")
    text = "mDice per test set (clean = protocolo limpo, sem TTA)\n\n" + "\n".join(table) + "\n"
    (Path(out_dir) / "summary.md").write_text(text)
    return text


def bench(out_dir, data_root, sets):
    """3-epoch B0 run; prints seconds per epoch (epoch 1 includes warm-up and is ignored)."""
    name = "bench_b0"
    exp = {"config": "configs/emcad_b0.yaml", "set": [*sets, "train.epochs=3"]}
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    run(train_cmd(name, exp, out_dir, data_root) + ["--no-test"], run_dir / "runner.log", echo=True)
    with open(run_dir / "log.csv") as f:
        t = [float(r["time_s"]) for r in csv.DictReader(f)]
    per_epoch = (t[-1] - t[0]) / (len(t) - 1)
    log(f"bench: {per_epoch:.1f} s/epoch for B0 at 352 px (Kaggle T4 reference: ~26 s/epoch)")
    log(f"bench: estimated 100-epoch B0 run: {per_epoch * 100 / 60:.0f} min + ~3 min of evaluation")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", help="experiment matrix YAML, e.g. experiments/round2.yaml")
    ap.add_argument("--only", nargs="+", help="run only these experiments (in matrix order)")
    ap.add_argument("--out", default="runs/round2", help="output directory (default runs/round2)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--jobs", type=int, default=1, help="experiments run concurrently on the same GPU")
    ap.add_argument("--force", action="store_true", help="retrain/re-evaluate even if results exist")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bench", action="store_true", help="only time a short B0 run")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    matrix = load_matrix(args.matrix)
    names = [n for n in matrix if not args.only or n in args.only]
    unknown = set(args.only or []) - set(matrix)
    if unknown:
        raise SystemExit(f"unknown experiments: {sorted(unknown)}; available: {list(matrix)}")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.bench:
        defaults = yaml.safe_load(open(args.matrix)).get("defaults", {}).get("set", [])
        return bench(args.out, args.data_root, defaults)
    if args.dry_run:
        for n in names:
            print(" ".join(map(str, train_cmd(n, matrix[n], args.out, args.data_root))))
            for v, flags in matrix[n]["evals"].items():
                print(f"    eval_{v}: {' '.join(map(str, flags))}")
        return
    if not args.summary_only:
        echo = args.jobs == 1
        failures = []

        def job(n):
            try:
                run_experiment(n, matrix[n], args.out, args.data_root, echo, args.force)
            except Exception as e:  # keep going with the rest of the matrix
                log(f"{n}: FAILED: {e}")
                failures.append(n)

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(job, names))
        if failures:
            log(f"failed: {failures}")
    text = summarize(args.out, names)
    print("\n" + text)
    log(f"summary written to {Path(args.out) / 'summary.md'}")
    if not args.summary_only and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
