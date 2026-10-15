"""Generate notebooks/vegalab_kaggle_gpu.ipynb, the GPU run of the Vega lab benchmark.

    python scripts/build_kaggle_notebook.py

The notebook only calls the `action-sentinel vega ...` CLI, so it runs exactly the code tested in
this repository. tests/unit/test_kaggle_notebook.py parses every command in it against the real CLI parser.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "vegalab_kaggle_gpu.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Vega Decision Intelligence Lab: GPU benchmark (Kaggle)

Runs the lab's controlled benchmark on a Kaggle GPU: **VegaML 0.8B and 4B** (adapters `off` and `auto`)
against the **same frozen backbone used as a plain language model** (option scoring) and a
**majority-class** baseline, on public tasks, with larger samples than the CPU run.

**Before running:** Settings → Accelerator → *GPU T4 x2* (or P100), Internet **on**. Optionally add a
Kaggle secret `HF_TOKEN` (the checkpoints and datasets are public; a token only avoids rate limits).

Everything here calls the `action-sentinel vega ...` CLI from the repository, so the code is the code
tested in the repo. Outputs (raw predictions, run metadata, metrics, reports) are zipped at the end for
download into `reports/vegalab/` locally.
"""),
    code("""
# ---------------- settings ----------------
REPO_URL = "https://github.com/premm25/action-sentinel.git"  # or attach the repo as a Kaggle dataset
REPO_DATASET_DIR = "/kaggle/input/action-sentinel"                  # used if REPO_URL is not reachable
MODELS = ["0.8b", "4b"]           # 4b needs ~9 GB GPU memory in fp16
N_TEST, N_CAL, N_DEV = 500, 200, 40
SEED = 20261015                   # same seed as the CPU run (samples differ from pub-v1 because n differs)
SYSTEMS = "majority,vega-off,vega-auto,backbone-optscore"
SPLITS = "test,calibration"
DATASET_VERSION = "pub-n%d-s%d" % (N_TEST, SEED)
WORK = "/kaggle/working"
"""),
    code("""
import os, subprocess, sys
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception:
    print("no HF_TOKEN secret; continuing anonymously")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo = os.path.join(WORK, "action-sentinel")
if not os.path.isdir(repo):
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, repo])
    if r.returncode != 0:
        assert os.path.isdir(REPO_DATASET_DIR), "set REPO_URL or attach the repository as a dataset"
        subprocess.run(["cp", "-r", REPO_DATASET_DIR, repo], check=True)
os.chdir(repo)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
"""),
    code("""
# torch with CUDA is preinstalled on Kaggle; do not reinstall it (that can break the CUDA build).
!pip install -q "vegaml==0.9.0" "transformers==5.19.0" pyarrow
!pip install -q --no-deps -e .
!pip install -q "pydantic>=2.7" pyyaml httpx
import torch, transformers, vegaml
print("torch", torch.__version__, "| transformers", transformers.__version__, "| vegaml", vegaml.__version__)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available(), "enable a GPU accelerator"
"""),
    md("## 1. Check installation and run one real decision"),
    code("""
!action-sentinel vega check
!action-sentinel vega decide --device cuda --adapters off --instructions "Which team should handle this?" --state "Invoice overdue, third notice" --option "billing=payments and invoices" --option "technical=product faults"
"""),
    md("## 2. Build the dataset (same builder and seed as the CPU run, larger samples)"),
    code("""
!action-sentinel vega data build --version {DATASET_VERSION} --root data/vegalab --n-test {N_TEST} --n-cal {N_CAL} --n-dev {N_DEV} --seed {SEED}
!action-sentinel vega data check --version {DATASET_VERSION} --root data/vegalab
"""),
    md("""
## 3. Inference

One run per checkpoint. VegaML (both adapter modes) shares one loaded model; the backbone baseline is
loaded only after VegaML is released. Runs resume if the session is interrupted.
"""),
    code("""
for model in MODELS:
    out = f"{WORK}/runs/{DATASET_VERSION}_{model}_cuda"
    !action-sentinel vega run --dataset data/vegalab/{DATASET_VERSION} --out {out} --systems {SYSTEMS} --splits {SPLITS} --model {model} --device cuda
"""),
    md("## 4. Metrics and reports (from the saved predictions only)"),
    code("""
for model in MODELS:
    out = f"{WORK}/runs/{DATASET_VERSION}_{model}_cuda"
    !action-sentinel vega analyze {out} --title "Vega lab, {model} on GPU, {DATASET_VERSION}"
    !action-sentinel vega failures {out} --system vega-off --kind confident-wrong --limit 10
"""),
    code("""
from IPython.display import Markdown, display
for model in MODELS:
    display(Markdown(open(f"{WORK}/runs/{DATASET_VERSION}_{model}_cuda/REPORT.md").read()))
"""),
    md("## 5. Package results for download"),
    code("""
!cp -r data/vegalab/{DATASET_VERSION} {WORK}/runs/dataset_{DATASET_VERSION}
!cd {WORK} && rm -f vegalab_results.zip && zip -qr vegalab_results.zip runs
print("Download /kaggle/working/vegalab_results.zip (Output tab), unzip into reports/vegalab/ in the repo.")
"""),
]


def main() -> None:
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"},
                                        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
