#!/usr/bin/env python3
"""Summarize HandPoseReplay / HandPoseMatrix JSONL output."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunStats:
    run_id: str
    name: str
    frames: int = 0
    frames_with_error: int = 0
    frames_with_observations: int = 0
    observations: int = 0
    joints: int = 0
    normal_conf: int = 0
    zero_conf: int = 0
    high_conf: int = 0
    negative_conf: int = 0
    nonfinite_conf: int = 0
    invalid_coords: int = 0
    duration_ms_total: float = 0.0
    errors_by_code: dict[str, int] = field(default_factory=dict)
    max_confidence: float | None = None
    anomaly_frames: list[dict[str, Any]] = field(default_factory=list)

    def add(self, record: dict[str, Any], anomaly_limit: int) -> None:
        self.frames += 1
        self.duration_ms_total += as_float(record.get("durationMs"), default=0.0)

        error = record.get("error")
        if error:
            self.frames_with_error += 1
            key = f"{error.get('domain', '?')}:{error.get('code', '?')}"
            self.errors_by_code[key] = self.errors_by_code.get(key, 0) + 1

        summary = record.get("summary") or {}
        obs_count = int(summary.get("observationCount") or len(record.get("observations") or []))
        joint_count = int(summary.get("jointCount") or 0)
        high_count = int(summary.get("outOfRangeHighConfidenceCount") or 0)
        invalid_count = int(summary.get("invalidCoordinateCount") or 0)

        if obs_count > 0:
            self.frames_with_observations += 1
        self.observations += obs_count
        self.joints += joint_count
        self.normal_conf += int(summary.get("normalConfidenceCount") or 0)
        self.zero_conf += int(summary.get("zeroConfidenceCount") or 0)
        self.high_conf += high_count
        self.negative_conf += int(summary.get("negativeConfidenceCount") or 0)
        self.nonfinite_conf += int(summary.get("nonFiniteConfidenceCount") or 0)
        self.invalid_coords += invalid_count

        frame_max_conf = max_confidence(record)
        if frame_max_conf is not None:
            self.max_confidence = (
                frame_max_conf
                if self.max_confidence is None
                else max(self.max_confidence, frame_max_conf)
            )

        if (error or high_count or invalid_count) and len(self.anomaly_frames) < anomaly_limit:
            self.anomaly_frames.append(
                {
                    "frameIndex": record.get("frameIndex"),
                    "sourceFrameIndex": record.get("sourceFrameIndex"),
                    "timestampSeconds": record.get("timestampSeconds"),
                    "error": error,
                    "outOfRangeHighConfidenceCount": high_count,
                    "invalidCoordinateCount": invalid_count,
                    "maxConfidence": frame_max_conf,
                }
            )

    def row(self) -> dict[str, Any]:
        error_rate = self.frames_with_error / self.frames if self.frames else 0.0
        high_rate = self.high_conf / self.joints if self.joints else 0.0
        mean_ms = self.duration_ms_total / self.frames if self.frames else 0.0
        return {
            "run_id": self.run_id,
            "name": self.name,
            "frames": self.frames,
            "errors": self.frames_with_error,
            "error_rate": error_rate,
            "observation_frames": self.frames_with_observations,
            "observations": self.observations,
            "joints": self.joints,
            "high_conf": self.high_conf,
            "high_conf_rate": high_rate,
            "zero_conf": self.zero_conf,
            "invalid_coords": self.invalid_coords,
            "max_confidence": self.max_confidence,
            "mean_ms": mean_ms,
            "errors_by_code": self.errors_by_code,
        }


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value == "NaN":
            return math.nan
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        try:
            return float(value)
        except ValueError:
            return default
    return default


def max_confidence(record: dict[str, Any]) -> float | None:
    best: float | None = None
    for obs in record.get("observations") or []:
        for joint in obs.get("joints") or []:
            conf = as_float(joint.get("confidence"))
            if conf is None or not math.isfinite(conf):
                continue
            best = conf if best is None else max(best, conf)
    return best


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
        else:
            paths.append(path)
    return paths


def iter_frame_records(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if record.get("recordType") == "frame" or "frameIndex" in record:
                    yield path, record


def config_name(record: dict[str, Any]) -> str:
    config = record.get("config") or {}
    return config.get("name") or record.get("runId") or "unknown"


def build_stats(paths: list[Path], anomaly_limit: int) -> dict[str, RunStats]:
    runs: dict[str, RunStats] = {}
    for _path, record in iter_frame_records(paths):
        run_id = str(record.get("runId") or config_name(record))
        stats = runs.get(run_id)
        if stats is None:
            stats = RunStats(run_id=run_id, name=config_name(record))
            runs[run_id] = stats
        stats.add(record, anomaly_limit=anomaly_limit)
    return runs


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "run",
        "frames",
        "errors",
        "err%",
        "obs_frames",
        "obs",
        "joints",
        "high_conf",
        "high%",
        "invalid_xy",
        "max_conf",
        "mean_ms",
    ]
    print("\t".join(headers))
    for row in rows:
        max_conf = row["max_confidence"]
        max_conf_text = "" if max_conf is None else f"{max_conf:.3f}"
        print(
            "\t".join(
                [
                    row["name"],
                    str(row["frames"]),
                    str(row["errors"]),
                    f"{row['error_rate'] * 100:.1f}",
                    str(row["observation_frames"]),
                    str(row["observations"]),
                    str(row["joints"]),
                    str(row["high_conf"]),
                    f"{row['high_conf_rate'] * 100:.1f}",
                    str(row["invalid_coords"]),
                    max_conf_text,
                    f"{row['mean_ms']:.2f}",
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSONL file(s) or directories")
    parser.add_argument(
        "--dump-anomalies",
        type=int,
        default=5,
        help="Keep and print up to N anomalous frames per run",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path for machine-readable aggregate summary",
    )
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing input(s): {', '.join(missing)}")

    runs = build_stats(paths, anomaly_limit=max(0, args.dump_anomalies))
    rows = [stats.row() for stats in runs.values()]
    rows.sort(key=lambda row: (-row["error_rate"], -row["high_conf_rate"], row["name"]))

    print_table(rows)

    if args.dump_anomalies:
        for stats in runs.values():
            if not stats.anomaly_frames:
                continue
            print()
            print(f"anomalies: {stats.name}")
            for anomaly in stats.anomaly_frames:
                print(json.dumps(anomaly, sort_keys=True))

    if args.summary_json:
        output = {
            "inputs": [str(path) for path in paths],
            "runs": rows,
        }
        Path(args.summary_json).write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
