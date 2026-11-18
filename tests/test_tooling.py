import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import kaggle_run  # noqa: E402


def test_check_secrets_blocks_token(tmp_path):
    bad, good = tmp_path / "bad.txt", tmp_path / "good.txt"
    bad.write_text("token = ghp_" + "a" * 36 + "\n")
    good.write_text("token = os.environ['HF_TOKEN']\n")
    script = str(ROOT / "scripts" / "check_secrets.sh")
    assert subprocess.run(["bash", script, str(bad)]).returncode == 1
    assert subprocess.run(["bash", script, str(good)]).returncode == 0


def test_parse_status():
    assert kaggle_run.parse_status('u/s has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert kaggle_run.parse_status('u/s has status "running"') == "running"


def test_scan_deletes_leaked_outputs(tmp_path):
    (tmp_path / "log.txt").write_text("hf_" + "x" * 34)
    (tmp_path / "ok.json").write_text("{}")
    bad = kaggle_run.scan_for_secrets(tmp_path)
    assert [b.name for b in bad] == ["log.txt"] and not (tmp_path / "log.txt").exists()


def test_apply_overrides():
    code = "A = 1\nB = 'x'\n    A = 5\n"
    assert kaggle_run.apply_overrides(code, ["A=2"]) == "A = 2\nB = 'x'\n    A = 5\n"
