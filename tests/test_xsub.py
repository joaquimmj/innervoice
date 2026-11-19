"""End-to-end smoke test of kaggle/xsub/xsub.py on synthetic MEG where each word type has its own spatial pattern."""
import json, os, subprocess, sys
from pathlib import Path

import h5py, numpy as np

ROOT = Path(__file__).resolve().parent.parent


def make_recording(root, sub, task, run, words, patterns, rng):
    n = 50 * (len(words) + 6)
    x = rng.normal(0, 1, (306, n)).astype(np.float32)
    rows = ["idx\twavile\tkind\tsegment\tsentenceidx\twordidx\tphonemeidx\ttimemeg\ttimeds\ttimechapter\ttimesentence\tduration"]
    for i, w in enumerate(words):
        t = 2.0 + i
        x[:, int(t * 50):int(t * 50) + 25] += 3 * patterns[w][:, None]
        rows.append(f"{i}\tseg\tword\t{w}\t{i // 6}\t{i % 6}\t\t{t}\t{t}\t0\t0\t0.2")
    (root / "cache").mkdir(parents=True, exist_ok=True)
    with h5py.File(root / "cache" / f"sub-{sub}_ses-1_task-{task}_run-{run}_preproc-abc.h5", "w") as f:
        f["data"] = x * 1e-12
    ev = root / "LibriBrain" / "serialized" / task / "derivatives" / "events"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / f"sub-{sub}_ses-1_task-{task}_run-{run}_events.tsv").write_text("\n".join(rows) + "\n")


def test_xsub_smoke(tmp_path):
    rng = np.random.default_rng(0)
    types = [f"w{i}" for i in range(12)]
    patterns = {w: rng.normal(0, 1, 306).astype(np.float32) for w in types + ["newword1", "newword2"]}
    for run in range(1, 5):
        make_recording(tmp_path / "a", 0, "Sherlock2", run, list(rng.choice(types, 120)), patterns, rng)
    for sub in (1, 5):
        for run in (1, 2):
            make_recording(tmp_path / "b", sub, "Sherlock1", run, list(rng.choice(types + ["newword1", "newword2"], 150)), patterns, rng)
    env = dict(os.environ, XSUB_INPUT=str(tmp_path), XSUB_OUT=str(tmp_path), XSUB_SMOKE="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, str(ROOT / "kaggle" / "xsub" / "xsub.py")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((tmp_path / "xsub_result.json").read_text())
    assert res["test_subjects"] == ["sub5"] and res["data"]["unseen_types"] == 2
    assert [(r["seed"], r["mode"]) for r in res["runs"]] == [(0, "sentence"), (0, "single"), (0, "single_multi"), (0, "timing")]
    for r in res["runs"]:
        top = r["per_subject"]["sub5"]["top5"]
        assert 0 <= top["bal_top10"] <= 1 and top["chance_top10"] == 1.0
        assert set(r["controls_test_subjects"]) == {"shift", "jitter", "noise"}
        agg = r["per_subject"]["sub5"]
        assert agg["agg2"]["n_groups"] <= agg["agg1"]["n_groups"] // 2 + 12 and 0 <= agg["agg2"]["bal_top1"] <= 1
    cal = res["runs"][1]["calibration"]
    assert set(cal["summary_test_subjects"]) == {"zero", "bn_k1", "input_k1", "full_k1"}
    assert "input_s3_k1" in cal["per_subject"]["sub5"] and cal["per_subject"]["sub5"]["text_overlap_5gram"] < 0.5
    assert set(res["runs"][2]["calibration"]["summary_test_subjects"]) == {"zero", "bn_k1"}
    assert (tmp_path / "model_seed0_single.pt").exists() and (tmp_path / "eval_embeddings_seed0_sentence.pt").exists()
