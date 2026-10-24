"""Shipped files (settings, rule packs, dataset): resolution, `init`, and the wheel's bundle list."""

from __future__ import annotations

import ast
from pathlib import Path

from action_sentinel import resources
from action_sentinel.config import load_settings
from tests.conftest import ROOT


def test_relative_paths_fall_back_to_the_shipped_files(tmp_path):
    shipped = resources.resolve(tmp_path, "examples/rule_packs/core.yaml")
    assert shipped == resources.bundled_root() / "examples/rule_packs/core.yaml" and shipped.is_file()
    mine = tmp_path / "examples/rule_packs/core.yaml"
    mine.parent.mkdir(parents=True)
    mine.write_text("rules: []", encoding="utf-8")
    assert resources.resolve(tmp_path, "examples/rule_packs/core.yaml") == mine  # a project's own copy wins
    assert resources.resolve(tmp_path, "var/sentinel.db") == tmp_path / "var/sentinel.db"  # data stays local


def test_a_named_profile_missing_here_is_the_shipped_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SENTINEL_CONFIG", raising=False)
    s = load_settings("config/sentinel-vegaml.yaml")
    assert s.active_providers == ["rules", "vegaml-head"]
    assert s.path("examples/rule_packs/core.yaml").is_file()


def test_shipped_settings_keep_data_in_the_current_directory(tmp_path, monkeypatch):
    bundled = tmp_path / "pkg" / "_bundled"
    (bundled / "config").mkdir(parents=True)
    (bundled / "config" / "sentinel.yaml").write_text("environment: dev\n", encoding="utf-8")
    monkeypatch.setattr(resources, "_PACKAGE", tmp_path / "pkg")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv("SENTINEL_CONFIG", raising=False)
    s = load_settings()
    assert Path(s.base_dir) == work and s.path(s.db_path) == work / "var" / "sentinel.db"


def test_init_copies_without_overwriting(tmp_path, capsys):
    from action_sentinel.cli import main
    assert main(["init", str(tmp_path)]) == 0
    core = tmp_path / "examples/rule_packs/core.yaml"
    assert core.is_file() and (tmp_path / "config/sentinel.yaml").is_file()
    core.write_text("edited", encoding="utf-8")
    assert main(["init", str(tmp_path)]) == 0
    assert core.read_text(encoding="utf-8") == "edited" and "kept" in capsys.readouterr().out
    assert main(["init", str(tmp_path), "--force"]) == 0
    assert core.read_text(encoding="utf-8") != "edited"


def test_setup_py_bundles_what_resources_expects():
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    [node] = [n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "BUNDLED"]
    assert [tuple(x) for x in ast.literal_eval(node.value)] == resources.BUNDLED
    for sub, pattern in resources.BUNDLED:
        assert list((ROOT / sub).glob(pattern)), sub
