#!/usr/bin/env python
"""Select one ordered calibration subset for both BFQ analysis and CWE search."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Expected a JSON or JSONL calibration file: {path}")
    if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
        raise ValueError("Calibration data must be a nonempty list of records")
    return records


def select_records(records: list[dict], n_samples: int, seed: int) -> tuple[list[dict], list[int]]:
    if not 0 < n_samples <= len(records):
        raise ValueError(f"Requested {n_samples} distinct samples from {len(records)} records")
    indices = list(range(len(records)))
    np.random.default_rng(seed=seed).shuffle(indices)
    selected_indices = indices[:n_samples]
    return [records[index] for index in selected_indices], selected_indices


def save_manifest(source: Path, output: Path, n_samples: int, seed: int) -> Path:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output calibration paths must differ")
    records, indices = select_records(load_records(source), n_samples, seed)
    payload = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    payload_hash = hashlib.sha256(payload).hexdigest()
    metadata = {
        "source": str(source),
        "source_sha256": source_hash,
        "selected_sha256": payload_hash,
        "seed": seed,
        "n_samples": n_samples,
        "source_indices": indices,
    }
    metadata_path = output.with_name(output.stem + ".meta.json")
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError(f"Existing calibration manifest differs: {output}")
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise FileExistsError(f"Existing calibration metadata differs: {metadata_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(payload)
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = save_manifest(args.source, args.output, args.n_samples, args.seed)
    print(f"[BFQ] ordered calibration manifest: {output} ({args.n_samples} samples)")


if __name__ == "__main__":
    main()
