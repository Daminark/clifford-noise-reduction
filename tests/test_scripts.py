"""The experiment scripts, run on their small grids."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_regret_quick_run_records_every_point_and_resumes(tmp_path):
    regret = _load("regret")
    path = tmp_path / "regret.json"
    regret.main(["--quick", "--output", str(path), "--workers", "1"])
    data = json.loads(path.read_text(encoding="utf-8"))
    config = regret.QUICK
    cases = list(regret.cases(config))
    points = len(cases) * len(config.ts) * len(config.rs)
    assert len(data["grid"]) == points
    assert len(data["choices"]) == len(cases) * len(config.ceilings)
    for record in data["grid"].values():
        assert len(record["accepted"]) == config.draws
    assert data["direct_reestimates"].keys() == data["direct"].keys()
    for key, choice in data["choices"].items():
        case = key.rsplit("/", 1)[0]
        assert 1 <= len(choice["measured"]) <= config.top
        picks = [choice["bound"], *choice["measured"]]
        for pick in [*picks, choice["rule"], choice["rule_by_bound"]]:
            if isinstance(pick, list):
                t, r = pick
                assert regret._key(case, t, r) in data["reestimates"]
            else:
                assert pick in (regret.DIRECT, None)
        assert None not in picks
    assert data["runs"][0]["provenance"]["packages"]["stim"]
    assert "finished" in data["runs"][0]

    regret.main(["--quick", "--output", str(path), "--workers", "1"])
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["grid"] == data["grid"]
    assert again["reestimates"] == data["reestimates"]
    assert len(again["runs"]) == 2


@pytest.mark.parametrize(
    ("size", "n", "r"), [(5, 3, 0), (6, 3, 1), (29, 3, 3), (886, 25, 5)]
)
def test_paper_r_is_the_floor_of_log2_of_size_over_n(size, n, r):
    assert _load("regret").paper_r(size, n) == r


def test_regret_refuses_a_file_made_with_other_settings(tmp_path):
    regret = _load("regret")
    path = tmp_path / "regret.json"
    regret.main(["--quick", "--output", str(path), "--workers", "1"])
    data = json.loads(path.read_text(encoding="utf-8"))
    data["config"]["draws"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="other settings"):
        regret.main(["--quick", "--output", str(path), "--workers", "1"])
