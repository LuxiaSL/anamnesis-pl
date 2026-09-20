from test_floor_reuse import module
from extraction_perf_batch_plan import exact_batches
from anamnesis.extraction.batch_layout import ReplaySpan, pack_spans

import pytest


@pytest.fixture
def schedule(tmp_path):
    rows = [
        dict(key=[str(i), "A", 0, "h1"], tokens=[i, 1, 2, 3], start=2, end=4)
        for i in range(3)
    ]
    plans = exact_batches(rows)
    module.write_json(tmp_path / "batch-plan.json", plans)
    pin = dict(
        batch_plan_sha256=module.sha(tmp_path / "batch-plan.json"),
        pairing_policy="exact",
    )
    receipts = {}
    for batch, plan in enumerate(plans):
        spans = tuple(
            ReplaySpan(tuple(rows[i]["tokens"]), 2, 4) for i in plan["indices"]
        )
        digest = pack_spans(spans, 4).composition_sha256
        for slot, index in enumerate(plan["emitted_indices"]):

            def meta(phase):
                return dict(
                    batch_size=2,
                    batch_row=slot,
                    batch_composition_sha256=digest,
                    batch_invocation_id=f"{batch}/{phase}",
                )

            record = dict(
                batch_index=batch,
                emitted_rows_in_batch=len(plan["emitted_indices"]),
                filler_rows=plan["filler_count"],
                filler_identity_exact=True,
                telemetry=meta(0),
                repeat_telemetry=meta(1),
                seconds_per_row=2 / len(plan["emitted_indices"]),
                batch_seconds=2.0,
                batch_repeat_seconds=2.0,
            )
            receipts[tuple(rows[index]["key"])] = (record, None)
    return tmp_path, pin, rows, receipts


def test_complete_schedule_and_filler_cost_verified(schedule):
    result = module.validate_batch_schedule(*schedule)
    assert result["batches"] == 2 and result["fillers"] == 1
    assert result["all_reference_rows_emitted_once"]


def test_singleton_cannot_claim_divided_runtime(schedule):
    record, _ = schedule[3][tuple(schedule[2][-1]["key"])]
    record["seconds_per_row"] = 1.0
    with pytest.raises(ValueError, match="overhead hidden"):
        module.validate_batch_schedule(*schedule)


def test_composition_cannot_change_between_replays(schedule):
    record, _ = schedule[3][tuple(schedule[2][0]["key"])]
    record["repeat_telemetry"]["batch_composition_sha256"] = "changed"
    with pytest.raises(ValueError, match="composition mismatch"):
        module.validate_batch_schedule(*schedule)
