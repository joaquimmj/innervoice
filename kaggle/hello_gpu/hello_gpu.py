"""Phase 0 smoke test on Kaggle: GPU, torch, private-repo clone, Hugging Face download.
Writes /kaggle/working/hello_gpu.json. Never prints secret values."""
import json, os, shutil, subprocess

import torch

res = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
res["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
if res["cuda_available"]:
    res["gpus"] = [{"name": torch.cuda.get_device_name(i),
                    "vram_GiB": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
                    "bf16": torch.cuda.is_bf16_supported()} for i in range(torch.cuda.device_count())]
res["disk_free_GiB"] = round(shutil.disk_usage("/kaggle/working").free / 2**30, 1)
res["ram_GiB"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)

try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    gh, hf = sec.get_secret("GITHUB_TOKEN"), sec.get_secret("HF_TOKEN")
except Exception as e:  # report the failure type only
    gh = hf = None
    res["secrets_error"] = type(e).__name__

if gh:
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", "dev",
                        f"https://x-access-token:{gh}@github.com/joaquimmj/nervegear.git", "/kaggle/working/repo"],
                       capture_output=True, text=True)
    res["repo_clone_ok"] = r.returncode == 0
    shutil.rmtree("/kaggle/working/repo", ignore_errors=True)  # keep token-bearing .git/config out of outputs

if hf:
    from huggingface_hub import HfApi, hf_hub_download
    try:
        hf_hub_download("pnpl/MEG-XL", "README.md", token=hf, local_dir="/tmp/hf")
        files = HfApi(token=hf).list_repo_tree("pnpl/LibriBrain", repo_type="dataset", recursive=False)
        res["hf_ok"], res["libribrain_top_level"] = True, sorted(f.path for f in files)
    except Exception as e:
        res["hf_ok"], res["hf_error"] = False, f"{type(e).__name__}: {str(e)[:200]}"

json.dump(res, open("/kaggle/working/hello_gpu.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "nvidia_smi"}, indent=1))
