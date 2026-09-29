import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_bfq_calibration_manifest import load_records, save_manifest, select_records


class CalibrationManifestTests(unittest.TestCase):
    def test_selection_matches_original_cwe_order(self):
        records = [{"id": index} for index in range(100)]
        selected, indices = select_records(records, n_samples=64, seed=42)
        original_cwe_order = list(records)
        np.random.default_rng(seed=42).shuffle(original_cwe_order)
        self.assertEqual(selected, original_cwe_order[:64])
        self.assertEqual([row["id"] for row in selected], indices)
        self.assertEqual(len(set(indices)), 64)

    def test_manifest_and_metadata_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(10)))
            output = root / "selected.json"
            save_manifest(source, output, n_samples=5, seed=42)
            first = load_records(output)
            save_manifest(source, output, n_samples=5, seed=42)
            self.assertEqual(first, load_records(output))
            metadata = json.loads((root / "selected.meta.json").read_text())
            self.assertEqual(metadata["n_samples"], 5)
            self.assertEqual([row["id"] for row in first], metadata["source_indices"])
            with self.assertRaises(FileExistsError):
                save_manifest(source, output, n_samples=5, seed=7)

    def test_selection_rejects_repeated_samples(self):
        with self.assertRaises(ValueError):
            select_records([{"id": 1}], n_samples=2, seed=42)

    def test_qe_and_cwe_loaders_read_identical_order(self):
        try:
            from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
            from scripts.analyze_calib_quantization_effect import _load_calibration_examples
        except ImportError as exc:
            self.skipTest(f"Optional model dependencies unavailable: {exc}")

        class IdentityModel:
            def preprocess_data(self, images, item):
                return {"id": item["id"]}

            def data_collator(self, samples):
                return samples

            def generate_input(self, samples):
                return samples, {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps([{"id": index} for index in range(12)]))
            manifest = root / "selected.json"
            save_manifest(source, manifest, n_samples=8, seed=42)
            model = IdentityModel()
            qe_samples = _load_calibration_examples(model, str(manifest), str(root), 8, 42, shuffle=False)
            cwe_samples, _ = get_multimodal_calib_dataset(
                data_path=str(manifest), image_folder=str(root), model=model,
                n_samples=8, shuffle=False, require_exact_n_samples=True,
            )
            self.assertEqual(qe_samples, cwe_samples)


if __name__ == "__main__":
    unittest.main()
