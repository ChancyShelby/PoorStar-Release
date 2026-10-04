#!/usr/bin/env python
"""Run one owned actor or learner league job."""

import argparse
from dataclasses import asdict
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    from poorstar.league.contracts import WorkerRequest
    from poorstar.league.ownership import read_document, write_document
    from poorstar.training.job_worker import execute_actor_request

    request = WorkerRequest(**read_document(args.request_json))
    expected = Path(request.output_dir) / "worker_result.json"
    if args.result_json.resolve() != expected.resolve() or expected.exists():
        parser.error("fresh result path must belong to this exact claimed attempt")
    result = execute_actor_request(request)
    write_document(expected, asdict(result), immutable=True)
    return 0 if result.status in ("COMPLETE", "TRUNCATED") and result.owned_sessions_closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
