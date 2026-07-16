"""Batch-send prediction requests to the drift_detector server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

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

_ALL_COLUMNS = [*_FEATURE_COLUMNS, "target"]


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


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = Report(
        [
            DataDriftPreset(num_stattest="t_test"),
        ]
    )
    report.run(
        reference_data=reference[_ALL_COLUMNS],
        current_data=current[_ALL_COLUMNS],
    )
    path = output_dir / f"{name}.html"
    report.save_html(str(path))
    result = report.as_dict()
    share = result["metrics"][0]["result"]["share_of_drifted_columns"]
    print(f"    [drift] share_of_drifted_columns={share:.3f}  ->  {path}")


def run(args: argparse.Namespace) -> None:
    url = f"{args.server.rstrip('/')}/predict/batch"
    data_files = find_data_files(args.data_dir)

    if not data_files:
        print(f"No data files found under {args.data_dir}/data/drift_*/data.csv")
        return

    reference = pd.read_csv(args.ref_data)
    print(f"Reference: {args.ref_data} ({len(reference)} rows)")
    print(f"Found {len(data_files)} data file(s)")
    total_predictions = 0

    for path in data_files:
        print(f"\n--- {path.parent.name} ({path}) ---")
        df = pd.read_csv(path)
        instances = df[_FEATURE_COLUMNS].to_dict(orient="records")
        n_rows = len(instances)
        print(f"  {n_rows} rows, batch size {args.batch_size}")

        report_count = 0

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

            batch_num = i // args.batch_size + 1
            if (
                args.drift_every > 0
                and batch_num >= args.drift_after
                and (batch_num - args.drift_after) % args.drift_every == 0
            ):
                current = df.iloc[: i + args.batch_size]
                report_count += 1
                name = f"{path.parent.name}_report_{report_count}"
                generate_drift_report(
                    reference,
                    current,
                    Path(args.drift_output),
                    name,
                )

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
    parser.add_argument(
        "--drift-every",
        type=int,
        default=0,
        metavar="N",
        help="Run drift report every N batches (0 to disable, default: 0).",
    )
    parser.add_argument(
        "--drift-after",
        type=int,
        default=1,
        metavar="N",
        help="Start drift reports after this many batches (default: 1).",
    )
    parser.add_argument(
        "--drift-output",
        default="reports",
        help="Directory to save drift HTML reports (default: reports).",
    )
    parser.add_argument(
        "--ref-data",
        default="data/train.csv",
        help="Path to reference (training) data CSV (default: data/train.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
