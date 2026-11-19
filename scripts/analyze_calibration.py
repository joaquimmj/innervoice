"""Paired calibration gains with bootstrap CIs, from an xsub_result.json.

Usage: python scripts/analyze_calibration.py results/<run>/metrics.json [--vs A B ...] [--metrics top50 top250]
For every calibration setting: mean over seeds per subject, then mean gain vs zero-shot over subjects with a
95 % bootstrap CI over subjects and the number of subjects improved. --vs A B prints paired differences A - B.
Only subjects that have the setting are used (a subject is skipped at K minutes when its recording is too short).
"""
import argparse, json

import numpy as np


def per_subject(runs, subjects, name, metric):
    """{subject: mean over seeds} for subjects present in every run."""
    out = {}
    for k in subjects:
        vals = [r["calibration"]["per_subject"].get(k, {}).get(name, {}) for r in runs]
        if all(isinstance(v, dict) and metric in v for v in vals):
            out[k] = float(np.mean([v[metric] for v in vals]))
    return out


def summarise(diff, n_boot=10000, seed=0):
    x = np.array(list(diff.values()))
    b = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                improved=int((x > 0).sum()), n=len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_json")
    ap.add_argument("--metrics", nargs="+", default=["top50", "top250"])
    ap.add_argument("--vs", nargs=2, action="append", default=[], metavar=("A", "B"))
    a = ap.parse_args()
    d = json.load(open(a.metrics_json))
    runs = [r for r in d["runs"] if r.get("calibration")]
    subjects = d["test_subjects"]
    names = sorted({n for r in runs for row in r["calibration"]["per_subject"].values() for n in row
                    if isinstance(row[n], dict) and n != "zero"})
    fmt = lambda s: f"{100 * s['mean']:+6.2f} [{100 * s['lo']:+6.2f}, {100 * s['hi']:+6.2f}] {s['improved']}/{s['n']}"
    for m in a.metrics:
        zero = per_subject(runs, subjects, "zero", m)
        print(f"\n{m}: zero-shot {100 * np.mean(list(zero.values())):.2f} ({len(zero)} subjects, {len(runs)} seeds)")
        for n in names:
            v = per_subject(runs, subjects, n, m)
            if v:
                print(f"  {n:32s} abs {100 * np.mean(list(v.values())):6.2f}  gain {fmt(summarise({k: v[k] - zero[k] for k in v}))}")
        for mode, ens in d.get("ensembles", {}).items():
            rows = ens["per_subject"]
            for name in sorted({n for r in rows.values() for n in r}):
                v = {k: rows[k][name][m] for k in subjects if k in rows and name in rows[k] and m in rows[k][name]}
                if v:
                    g = summarise({k: v[k] - zero[k] for k in v if k in zero})
                    print(f"  ensemble[{mode}] {name:18s} abs {100 * np.mean(list(v.values())):6.2f}  gain vs single-model zero {fmt(g)}")
        for x, y in a.vs:
            vx, vy = per_subject(runs, subjects, x, m), per_subject(runs, subjects, y, m)
            common = sorted(set(vx) & set(vy))
            if common:
                print(f"  {x} - {y}: {fmt(summarise({k: vx[k] - vy[k] for k in common}))}")


if __name__ == "__main__":
    main()
