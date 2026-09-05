import json
from pathlib import Path

import pytest

from hud_dropbear.cli import parser, selected_tasks
from hud_dropbear.contract import TASK_SUITES

SOURCE = Path(__file__).resolve().parents[1] / "tasksets/five-suites.json"


def test_portable_hud_taskset_selects_distinct_tasks_per_suite():
    rows = selected_tasks(parser().parse_args(["--taskset", str(SOURCE)]))
    assert [row.id for row in rows] == list(TASK_SUITES)
    assert [row.args["task_id"] for row in rows] == [0, 0, 0, 0, 50]
    assert all(row.args["max_steps"] == 600 for row in rows)
    assert rows[-1].columns["task_name"] == TASK_SUITES["libero_90"][50]


@pytest.mark.parametrize("field,value", [("task_id", True), ("max_steps", 1), ("seed", 3)])
def test_taskset_cannot_misreport_its_evaluation_contract(tmp_path, field, value):
    data = json.loads(SOURCE.read_text())
    data[0]["args"][field] = value
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        selected_tasks(parser().parse_args(["--taskset", str(path)]))


def test_taskset_rejects_name_and_numeric_id_drift(tmp_path):
    data = json.loads(SOURCE.read_text())
    data[0]["columns"]["task_name"] = "a different task"
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="task_name"):
        selected_tasks(parser().parse_args(["--taskset", str(path)]))
