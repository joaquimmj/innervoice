"""CPU job: turn LibriBrain (sub-0, 91 serialised h5, 55 GiB @250 Hz) into MEG-XL's 50 Hz cache (~11 GiB float32).

Streams one recording at a time (download -> MEG-XL preprocessing -> cache -> delete) so it fits Kaggle's disk.
Output tree (usable as `data.root` / `data.cache_dir` of MEG-XL's eval scripts by a later GPU job):
  LibriBrain/meg_sensors_information.json
  LibriBrain/serialized/<task>/derivatives/serialised/*.h5   attribute-only stubs (make MEG-XL pick the cache)
  LibriBrain/serialized/<task>/derivatives/events/*.tsv
  cache/*.h5                                                 same names/hash as MEG-XL's own cache
Differences from MEG-XL's cache: float32 instead of float64 (disk limit). Everything else is their code.
"""
import json, os, re, shutil, subprocess, sys, time, traceback

MEGXL_COMMIT = "97da04171d22e8fe4822cfeb3047a57ecf86b318"
REPO = "pnpl/LibriBrain"
SUBJECT_RE = r"sub-0"  # regex on the subject part of the file name
TASKS = None  # None = every task; else a list like ["Sherlock1"]
SENSOR_REPO = "pnpl/LibriBrain"  # raw FIF used for sensor geometry (same scanner for all subjects)
L_FREQ, H_FREQ, SFREQ = 0.1, 40.0, 50.0
LIMIT = int(os.environ.get("PREP_LIMIT", "0"))  # >0: only the first N recordings (smoke test)
W = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
TMP = "/tmp/dl"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning", "einops", "mne", "h5py"], check=True)
src = "/tmp/MEG-XL"
if not os.path.isdir(src):
    subprocess.run(["git", "clone", "-q", "https://github.com/neural-processing-lab/MEG-XL.git", src], check=True)
    subprocess.run(["git", "-C", src, "checkout", "-q", MEGXL_COMMIT], check=True)
sys.path.insert(0, src)

import h5py, mne, numpy as np
from huggingface_hub import HfApi, hf_hub_download
# import the module file directly: brainstorm.data.__init__ pulls in every dataset class
import importlib.util
spec = importlib.util.spec_from_file_location("megxl_prep", f"{src}/brainstorm/data/preprocessing.py")
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)

mne.set_log_level("ERROR")
root, cache = f"{W}/LibriBrain", f"{W}/cache"
os.makedirs(cache, exist_ok=True)
files = [f.path for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True)]
h5s = sorted(p for p in files if p.endswith(".h5") and "/serialised/" in p and re.search(rf"/({SUBJECT_RE})_", p)
             and (TASKS is None or p.split("/")[0] in TASKS))
if LIMIT:
    h5s = h5s[:LIMIT]


def fetch(path, repo=None):
    return hf_hub_download(repo or REPO, path, repo_type="dataset", local_dir=TMP)


# 1. Sensor geometry (position + orientation + coil type) from one raw FIF header; not shipped as JSON on the Hub.
sensor_files = [f.path for f in HfApi().list_repo_tree(SENSOR_REPO, repo_type="dataset", recursive=True)]
fif = next(p for p in sorted(sensor_files) if p.endswith("_meg.fif") and "proc-bads+headpos+sss+notch+bp+ds" in p)
info = mne.io.read_info(fetch(fif, SENSOR_REPO))
sensors = [dict(ch_name=c["ch_name"], loc=[float(x) for x in c["loc"]], coil_type=int(c["coil_type"]))
           for c in info["chs"] if c["ch_name"].startswith("MEG")]
os.makedirs(root, exist_ok=True)
json.dump(sensors, open(f"{root}/meg_sensors_information.json", "w"))
shutil.rmtree(TMP, ignore_errors=True)
xyzdir, types = pp.load_libribrain_sensors(f"{root}/meg_sensors_information.json")
print(f"sensors: {len(sensors)} from {fif}", flush=True)

# 2. Recordings.
log, t0 = [], time.time()
for i, path in enumerate(h5s):
    task, name = path.split("/")[0], os.path.basename(path)
    parts = name.replace(".h5", "").split("_")
    sub, ses, run = parts[0], parts[1], parts[3]
    ev_name = f"{sub}_{ses}_task-{task}_{run}_events.tsv"
    ev_src = f"{task}/derivatives/events/{ev_name}"
    cache_path = f"{cache}/{sub}_{ses}_task-{task}_{run}_preproc-{pp.compute_preproc_hash(L_FREQ, H_FREQ, SFREQ, 'MEG_only')}.h5"
    rec = dict(path=path, ok=False)
    try:
        assert ev_src in files, f"no events file {ev_src}"
        local = fetch(path)
        with h5py.File(local, "r") as f:
            attrs = {k: f.attrs[k] for k in ("sample_frequency", "highpass_cutoff", "lowpass_cutoff",
                                             "channel_names", "channel_types") if k in f.attrs}
        missing = [c for c in attrs["channel_names"].split(", ") if c.startswith("MEG") and c not in xyzdir]
        assert not missing, f"channels without geometry: {missing[:5]}"
        raw = pp.preprocess_libribrain_h5(local, xyzdir, types, l_freq=L_FREQ, h_freq=H_FREQ, target_sfreq=SFREQ,
                                          channel_filter=lambda x: x.startswith("MEG"))
        get64 = raw.get_data
        raw.get_data = lambda *a, **k: get64(*a, **k).astype(np.float32)  # the only deviation: halve disk use
        pp.cache_preprocessed(raw, __import__("pathlib").Path(cache_path),
                              dict(subject=sub, session=ses, task=task, run=run, dataset="libribrain"),
                              l_freq=L_FREQ, h_freq=H_FREQ, target_sfreq=SFREQ, channel_filter_name="MEG_only")
        for d in ("serialised", "events"):
            os.makedirs(f"{root}/serialized/{task}/derivatives/{d}", exist_ok=True)
        with h5py.File(f"{root}/serialized/{task}/derivatives/serialised/{name}", "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
        shutil.copy(fetch(ev_src), f"{root}/serialized/{task}/derivatives/events/{ev_name}")
        with h5py.File(cache_path, "r") as f:
            rec.update(ok=True, n_channels=int(f["data"].shape[0]), minutes=round(f["data"].shape[1] / SFREQ / 60, 1),
                       finite=bool(np.isfinite(f["data"][:, :500]).all()))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)
    log.append(rec)
    if i == 0 and not rec["ok"]:
        sys.exit("first recording failed: aborting instead of failing 91 times")
    print(f"[{i + 1}/{len(h5s)}] {(time.time() - t0) / 60:.0f} min {rec}", flush=True)
    json.dump(dict(megxl_commit=MEGXL_COMMIT, recordings=log,
                   hours=round(sum(r.get("minutes", 0) for r in log) / 60, 2),
                   cache_GiB=round(sum(os.path.getsize(f"{cache}/{x}") for x in os.listdir(cache)) / 2**30, 2),
                   disk_free_GiB=round(shutil.disk_usage(W).free / 2**30, 1)),
              open(f"{W}/prep_summary.json", "w"), indent=1)
sys.exit(0 if log and all(r["ok"] for r in log) else 1)
