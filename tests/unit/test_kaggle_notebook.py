"""The Kaggle notebook must stay in sync with the CLI: every command it runs must parse."""

from __future__ import annotations

import json
import re
import shlex

from action_sentinel.cli import build_parser
from tests.conftest import ROOT

NB = ROOT / "notebooks" / "vegalab_kaggle_gpu.ipynb"
VARS = {"DATASET_VERSION": "pub-n500-s20261015", "N_TEST": "500", "N_CAL": "200", "N_DEV": "40", "SEED": "20261015",
        "SYSTEMS": "majority,vega-off,vega-auto,backbone-optscore", "SPLITS": "test,calibration", "model": "0.8b",
        "out": "/kaggle/working/runs/x", "WORK": "/kaggle/working"}


def test_notebook_is_generated_and_current():
    import runpy
    before = NB.read_text(encoding="utf-8")
    runpy.run_path(str(ROOT / "scripts" / "build_kaggle_notebook.py"), run_name="__main__")
    assert NB.read_text(encoding="utf-8") == before, "notebook is stale: run scripts/build_kaggle_notebook.py"


def test_every_cli_command_in_notebook_parses():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cmds = []
    for cell in nb["cells"]:
        for line in "".join(cell["source"]).splitlines():
            line = line.strip()
            if line.startswith("!action-sentinel"):
                line = re.sub(r"\{(\w+)\}", lambda m: VARS[m.group(1)], line)
                cmds.append(shlex.split(line[len("!action-sentinel"):]))
    assert len(cmds) >= 7
    parser = build_parser()
    for argv in cmds:
        args = parser.parse_args(argv)
        assert args.cmd == "vega" and callable(args.fn)
    assert any(a[:2] == ["vega", "run"] and "--device" in a and "cuda" in a for a in cmds)
