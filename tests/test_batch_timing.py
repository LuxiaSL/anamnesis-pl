import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extraction_perf_batch_timing import summarize


def row(key, batch, count, fillers, seconds, single=None):
    return dict(
        key=[key],
        batch_index=batch,
        lane="lane",
        emitted_rows_in_batch=count,
        filler_rows=fillers,
        filler_identity_exact=True,
        batch_seconds=seconds,
        batch_repeat_seconds=seconds,
        seconds_per_row=seconds / count,
        single_seconds=single,
    )


def test_filler_is_charged_and_batch_calls_not_double_counted():
    result = summarize(
        [row("a", 0, 2, 0, 2, 1.2), row("b", 0, 2, 0, 2, 1.3), row("c", 1, 1, 1, 2)]
    )
    assert result["first_call_seconds"] == 4
    assert result["first_call_seconds_per_row"] == 4 / 3
    assert result["fillers"] == 1
    assert result["paired_sample_speedup"] == 1.25
    assert result["paired_sample_rows"] == 2
    assert result["paired_sample_by_filler"]["no_filler"]["speedup"] == 1.25


def test_filler_pairing_is_separately_priced():
    result = summarize([row("a", 0, 1, 1, 2, 1.1)])
    assert result["paired_sample_by_filler"]["with_filler"]["speedup"] == 0.55


@pytest.mark.parametrize(
    "change",
    [
        {"seconds_per_row": 1},
        {"batch_seconds": float("nan")},
        {"filler_identity_exact": False},
        {"emitted_rows_in_batch": 2},
    ],
)
def test_bad_receipt_rejected(change):
    record = row("a", 0, 1, 1, 2)
    record.update(change)
    with pytest.raises(ValueError):
        summarize([record])


def test_partial_paired_batch_rejected():
    with pytest.raises(ValueError, match="partial"):
        summarize([row("a", 0, 2, 0, 2, 1), row("b", 0, 2, 0, 2)])
