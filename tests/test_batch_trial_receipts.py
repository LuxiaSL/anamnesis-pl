import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "batch_trial", scripts / "extraction_perf_batch_trial.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_numpy_comparisons_are_json_safe_without_loosening_gate():
    for distance, bound, error, expected in (
        (0.01, 0.1, 0.01, True),
        (0.11, 0.1, 0.01, False),
        (0.01, 0.1, 0.11, False),
    ):
        value = module.feature_gate(
            np.float64(distance), np.float64(bound), np.float32(error)
        )
        assert type(value) is bool
        assert json.loads(json.dumps({"G1": value}))["G1"] is expected


def test_exact_shape_control_keeps_failing_row_and_has_no_padding():
    rows = [dict(key=[f"d{i}", "paraphrase_own", 0, "h1"], start=length, end=length+128)
            for length in (263,264,265,266,297,299) for i in range(3)]
    failed=dict(key=["s000","paraphrase_own",0,"h1"],start=297,end=425)
    rows.append(failed)
    selected=module.select_rows(rows,"exact")
    assert selected[0] == failed
    assert len(selected)==12
    assert len({tuple(r["key"])+(r["start"],) for r in selected})==12
    for a,b in zip(selected[::2],selected[1::2],strict=True):
        assert (a["start"],a["end"]) == (b["start"],b["end"])
