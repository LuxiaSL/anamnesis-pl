#!/usr/bin/env python3
"""Ask an ISOLATED Claude reader a question, with none of this project's context.

Why this exists: subagents launched from inside the repo inherit the project's
CLAUDE.md and memory. On 2026-09-18 that silently un-blinded a blind analogy run --
all three readers had "CURRENT FRONT: Rough paths" in their ambient context and
only one noticed. A blind test has to be blind in the ENVIRONMENT, not just in the
prompt. This script talks to the API directly: no system prompt, no tools, no repo,
one user turn carrying the brief and nothing else.

Usage::

    python3 blind_reader.py --brief BRIEF.md --n 3 --out DIR [--model claude-opus-5]

Requires ANTHROPIC_API_KEY in the environment. Each reader is an independent
single-turn request; readers cannot see each other.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

try:
    import anthropic
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    print(f"FATAL: missing dependency ({exc}). Need `anthropic` and `pydantic`.", file=sys.stderr)
    raise SystemExit(2) from exc

DEFAULT_MODEL: Final[str] = "claude-opus-5"
DEFAULT_MAX_TOKENS: Final[int] = 32_000


class ReaderResult(BaseModel):
    """One reader's answer plus everything needed to audit the call."""

    index: int = Field(ge=1)
    model: str
    ok: bool
    text: str = ""
    error: str | None = None
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class RunRecord(BaseModel):
    """The whole run, written next to the answers so provenance travels with them."""

    started_at: str
    model: str
    max_tokens: int
    brief_path: str
    brief_sha256: str
    n_readers: int
    isolation: str = (
        "direct Messages API call: no system prompt, no tools, no repository access, "
        "no project memory; one user turn containing the brief verbatim"
    )
    results: list[ReaderResult]


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def ask_one(client: "anthropic.Anthropic", brief: str, model: str, max_tokens: int, index: int) -> ReaderResult:
    """One reader. Streams so a long answer cannot hit an HTTP timeout.

    Thinking is deliberately left unset: on claude-opus-5 that runs adaptive
    thinking by default, and passing an explicit config would couple this script
    to one SDK generation.
    """
    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": brief}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.APIStatusError as exc:
        return ReaderResult(index=index, model=model, ok=False,
                            error=f"{type(exc).__name__}: status {exc.status_code}: {exc}")
    except anthropic.APIConnectionError as exc:
        return ReaderResult(index=index, model=model, ok=False, error=f"{type(exc).__name__}: {exc}")

    text = "\n".join(b.text for b in message.content if b.type == "text")
    return ReaderResult(
        index=index,
        model=model,
        ok=bool(text.strip()),
        text=text,
        error=None if text.strip() else "empty response",
        stop_reason=message.stop_reason,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brief", type=Path, required=True, help="file whose contents are the entire prompt")
    ap.add_argument("--n", type=int, default=3, help="independent readers (default 3)")
    ap.add_argument("--out", type=Path, required=True, help="directory for answers + run record")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = ap.parse_args()

    if not args.brief.is_file():
        print(f"FATAL: brief not found: {args.brief}", file=sys.stderr)
        return 2
    if args.n < 1:
        print("FATAL: --n must be >= 1", file=sys.stderr)
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("FATAL: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    brief = args.brief.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    results: list[ReaderResult] = []
    for i in range(1, args.n + 1):
        print(f"reader {i}/{args.n} ...", file=sys.stderr, flush=True)
        r = ask_one(client, brief, args.model, args.max_tokens, i)
        results.append(r)
        if r.ok:
            (args.out / f"reader-{i}.md").write_text(r.text, encoding="utf-8")
            print(f"  ok: {r.output_tokens} output tokens, stop={r.stop_reason}", file=sys.stderr)
        else:
            print(f"  FAILED: {r.error}", file=sys.stderr)

    record = RunRecord(
        started_at=datetime.now(timezone.utc).isoformat(),
        model=args.model,
        max_tokens=args.max_tokens,
        brief_path=str(args.brief),
        brief_sha256=sha256_of(args.brief),
        n_readers=args.n,
        results=results,
    )
    (args.out / "run.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.ok)
    print(f"\n{n_ok}/{args.n} readers answered. Record: {args.out / 'run.json'}", file=sys.stderr)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
