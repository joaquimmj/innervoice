"""Push a Kaggle kernel, wait for it, fetch its outputs, refuse to keep anything secret-looking.

Usage: python scripts/kaggle_run.py kaggle/<job> [--poll 60] [--timeout-h 12]
Needs KAGGLE_USERNAME and KAGGLE_KEY in the environment (never printed).
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET_RE = re.compile(r"(ghp_|github_pat_|gho_)[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}")
DONE = {"complete", "error", "cancelAcknowledged"}


def kaggle(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"kaggle {args[0]} {args[1]} failed: {r.stderr.strip()[-500:]}")
    return r.stdout


def parse_status(out):
    # CLI prints e.g. 'user/slug has status "KernelWorkerStatus.COMPLETE"' or 'has status "complete"'
    m = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', out)
    s = m.group(1).lower() if m else "unknown"
    return {"cancel_acknowledged": "cancelAcknowledged"}.get(s, s)


def scan_for_secrets(folder):
    bad = [p for p in Path(folder).rglob("*") if p.is_file()
           and SECRET_RE.search(p.read_bytes().decode("utf-8", "ignore"))]
    for p in bad:
        p.unlink()  # never let a leaked token reach git
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--timeout-h", type=float, default=12)
    a = ap.parse_args()
    for v in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        if not os.environ.get(v):
            sys.exit(f"{v} is not set (PARTIE 0.6)")
    meta = json.loads((Path(a.job_dir) / "kernel-metadata.json").read_text())
    ref = meta["id"] = meta["id"].replace("{KAGGLE_USERNAME}", os.environ["KAGGLE_USERNAME"])
    with tempfile.TemporaryDirectory() as tmp:  # push a copy so the tracked metadata keeps the placeholder
        shutil.copytree(a.job_dir, tmp, dirs_exist_ok=True)
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta))
        print(kaggle("kernels", "push", "-p", tmp).strip())
    t0, status = time.time(), "unknown"
    while time.time() - t0 < a.timeout_h * 3600:
        time.sleep(a.poll)
        status = parse_status(kaggle("kernels", "status", ref))
        print(f"[{(time.time() - t0) / 60:.0f} min] {status}", flush=True)
        if status in DONE:
            break
    out = ROOT / "results" / "raw" / Path(a.job_dir).name
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref, "-p", str(out))
    bad = scan_for_secrets(out)
    if bad:
        sys.exit(f"secret-like content found and deleted: {[str(b) for b in bad]}")
    print(f"status={status} outputs={out}")
    sys.exit(0 if status == "complete" else 1)


if __name__ == "__main__":
    main()
