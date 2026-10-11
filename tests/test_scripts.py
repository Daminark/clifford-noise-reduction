"""The experiment scripts, run on their small grids."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    # A script imports its siblings, as it would when run from scripts/.
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
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


def test_regret_outcomes_read_every_choice_from_fresh_estimates(tmp_path):
    regret = _load("regret")
    path = tmp_path / "regret.json"
    regret.main(["--quick", "--output", str(path), "--workers", "1"])
    data = json.loads(path.read_text(encoding="utf-8"))

    rows = regret.outcomes(data)

    assert len(rows) == len(data["choices"])
    for row in rows:
        key = regret._key(row["regime"], row["n"], row["circuit"])
        again = data["direct_reestimates"][key]
        assert row["direct"] == again["errors"] / again["shots"]
        for name in ("best", "best_pair", "bound", "rule", "rule_by_bound"):
            pick = row[f"{name}_pick"]
            if pick is None:
                assert math.isnan(row[name])
            elif pick == regret.DIRECT:
                assert row[name] == row["direct"]
            else:
                assert 0.0 <= row[name] <= 1.0


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


def test_sharpness_quick_run_records_every_point_and_resumes(tmp_path):
    regret = _load("regret")
    sharpness = _load("sharpness")
    source = tmp_path / "regret.json"
    path = tmp_path / "sharpness.json"
    regret.main(["--quick", "--output", str(source), "--workers", "1"])
    arguments = ["--quick", "--regret", str(source), "--output", str(path)]
    sharpness.main(arguments)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = json.loads(source.read_text(encoding="utf-8"))
    config = sharpness.QUICK

    chosen = {
        regret._key(
            row["regime"],
            row["n"],
            row["circuit"],
            row["family"],
            *row["best_pair_pick"],
        )
        for row in regret.outcomes(results)
        if isinstance(row["best_pair_pick"], list)
        and row["best_pair_pick"][1] >= 1
    }
    assert set(data["waterfall"]) == chosen
    for row in sharpness.waterfall(data, results):
        assert list(row["stages"]) == list(sharpness.STAGES)
        assert all(math.isfinite(value) for value in row["stages"].values())
        assert row["regions"]["bound"]["resource"] > 0
    assert len(data["positions"]) == (
        len(config.position_sizes)
        * config.position_circuits
        * len(sharpness.GENERATORS)
        * len(regret.FAMILIES)
    )
    for profile in data["positions"].values():
        assert len(profile["damaging"]) == config.bins + 1
        for damaging, caught in zip(
            profile["damaging"], profile["caught"], strict=True
        ):
            assert 0.0 <= caught <= damaging
    assert "finished" in data["runs"][0]

    sharpness.main(arguments)
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["waterfall"] == data["waterfall"]
    assert again["positions"] == data["positions"]
    assert len(again["runs"]) == 2


def test_sharpness_refuses_a_file_made_from_other_results(tmp_path):
    regret = _load("regret")
    sharpness = _load("sharpness")
    source = tmp_path / "regret.json"
    path = tmp_path / "sharpness.json"
    regret.main(["--quick", "--output", str(source), "--workers", "1"])
    arguments = ["--quick", "--regret", str(source), "--output", str(path)]
    sharpness.main(arguments)
    source.write_text(
        source.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="other regret results"):
        sharpness.main(arguments)


def test_payoff_quick_run_records_every_unit_and_resumes(tmp_path):
    payoff = _load("payoff")
    path = tmp_path / "payoff.json.gz"
    arguments = ["--quick", "--output", str(path), "--workers", "1"]
    payoff.main(arguments)
    data = payoff.load(path)
    config = payoff.QUICK

    cases = list(payoff.cases(config, data["sizes"]))
    circuits = sum(count for *_, count in cases)
    assert len(data["direct"]) == circuits
    assert len(data["units"]) == circuits * 2
    for unit in data["units"].values():
        assert unit["r"] >= 1
        for t in unit["choice"].values():
            if t is not None:
                overhead = unit["pilots"][str(t)][1]
                assert str(t) in unit["estimates"]
                assert overhead <= max(config.curve_budgets)
    rows = payoff.outcomes(data)
    budgets = {
        0: len(config.curve_budgets),
        1: len(config.map_budgets),
    }
    assert len(rows) == sum(
        2 * count * budgets[part] for part, *_, count in cases
    )
    for row in rows:
        assert math.isnan(row["rate"]) == (row["t"] is None)
    assert "finished" in data["runs"][0]

    payoff.main(arguments)
    again = payoff.load(path)
    assert again["units"] == data["units"]
    assert again["direct"] == data["direct"]
    assert len(again["runs"]) == 2


@pytest.mark.parametrize(
    ("size", "n", "r"), [(2, 3, 1), (5, 3, 1), (29, 3, 3), (625, 25, 4)]
)
def test_payoff_takes_at_least_one_check(size, n, r):
    assert _load("payoff").rule_r(size, n) == r


def test_dense_quick_run_records_every_unit_and_resumes(tmp_path):
    dense = _load("dense")
    path = tmp_path / "dense.json.gz"
    arguments = ["--quick", "--output", str(path), "--workers", "1"]
    dense.main(arguments)
    data = dense.payoff.load(path)
    config = dense.QUICK

    cases = sum(len(sizes) for sizes in data["sizes"].values())
    assert len(data["direct"]) == cases
    assert len(data["units"]) == cases * len(config.arms)
    for key, unit in data["units"].items():
        n = int(key.split("/")[1])
        assert unit["s"] == n * (n - 1) // 2
        assert unit["r"] >= 1
        for t in unit["choice"].values():
            if t is not None:
                assert str(t) in unit["estimates"]
                assert unit["pilots"][str(t)][1] <= max(config.budgets)
    rows = dense.outcomes(data)
    assert len(rows) == cases * len(config.arms) * len(config.budgets)
    for row in rows:
        assert math.isnan(row["rate"]) == (row["t"] is None)
    assert "finished" in data["runs"][0]

    dense.main(arguments)
    again = dense.payoff.load(path)
    assert again["units"] == data["units"]
    assert again["direct"] == data["direct"]
    assert len(again["runs"]) == 2


def test_dense_circuits_are_complete_graphs():
    circuit = _load("dense").complete_graph(4)

    pairs = [
        (
            instruction.targets_copy()[k].value,
            instruction.targets_copy()[k + 1].value,
        )
        for instruction in circuit
        for k in range(0, len(instruction.targets_copy()), 2)
    ]
    assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
