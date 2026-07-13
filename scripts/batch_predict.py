"""Batch-send prediction requests to the drift_detector server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

_FEATURE_COLUMNS = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_gaussian",
    "feature_lognormal",
    "feature_exponential",
    "category",
    "type",
]


def find_data_files(data_dir: str) -> list[Path]:
    return sorted(Path(data_dir).glob("drift_*/data.csv"))


def send_batch(
    url: str,
    instances: list[dict],
    timeout: float = 30.0,
) -> list[float]:
    payload = json.dumps({"instances": instances}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["predictions"]


def run(args: argparse.Namespace) -> None:
    url = f"{args.server.rstrip('/')}/predict/batch"
    data_files = find_data_files(args.data_dir)

    if not data_files:
        print(f"No data files found under {args.data_dir}/data/drift_*/data.csv")
        return

    print(f"Found {len(data_files)} data file(s)")
    total_predictions = 0

    for path in data_files:
        print(f"\n--- {path.parent.name} ({path}) ---")
        df = pd.read_csv(path)
        instances = df[_FEATURE_COLUMNS].to_dict(orient="records")
        n_rows = len(instances)
        print(f"  {n_rows} rows, batch size {args.batch_size}")

        for i in range(0, n_rows, args.batch_size):
            batch = instances[i : i + args.batch_size]
            try:
                preds = send_batch(url, batch, timeout=args.timeout)
                total_predictions += len(preds)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"  ERROR at batch {i}: {exc}")
                continue

            if args.progress and (i // args.batch_size) % args.progress == 0:
                print(f"  sent {min(i + args.batch_size, n_rows)}/{n_rows}")

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\nDone. {total_predictions} predictions sent.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-send prediction requests to the drift_detector server.",
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="Server base URL (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory containing drift_*/data.csv (default: data).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Rows per request (default: 100).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between batches (default: 0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=0,
        metavar="N",
        help="Print progress every N batches (0 to disable, default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
