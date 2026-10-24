"""Copies the shipped settings, rule packs, dataset and sample actions into the wheel.

They live in config/ and examples/ in the repository; a built package carries them in
action_sentinel/_bundled/ under the same relative paths (action_sentinel/resources.py finds them).
Everything else is configured in pyproject.toml.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
# Keep in step with action_sentinel.resources.BUNDLED (setup.py can't import the package while building it).
BUNDLED = [("config", "*.yaml"), ("examples/rule_packs", "*.yaml"), ("examples/evaluation_datasets", "*.jsonl"),
           ("examples/sample_actions", "*.json*")]


class BuildWithBundled(build_py):
    def run(self):
        super().run()
        dest = Path(self.build_lib) / "action_sentinel" / "_bundled"
        for sub, pattern in BUNDLED:
            for src in sorted((ROOT / sub).glob(pattern)):
                out = dest / sub / src.name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, out)


setup(cmdclass={"build_py": BuildWithBundled})
