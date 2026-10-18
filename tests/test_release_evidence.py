"""Validate published evidence bytes and the metadata references between them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs/det-fix_v1.1.0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseEvidenceTests(unittest.TestCase):
    def test_published_files_match_release_manifest(self):
        manifest = json.loads((EVIDENCE / "release_manifest.json").read_text())
        paths = [entry["release_path"] for entry in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        for entry in manifest["files"]:
            with self.subTest(path=entry["release_path"]):
                self.assertEqual(sha(ROOT / entry["release_path"]), entry["release_sha256"])

    def test_definition_and_summary_hashes_match_published_files(self):
        definition = json.loads((EVIDENCE / "definition/experiment.json").read_text())
        self.assertTrue(definition["publication_view"])
        reference = definition["challenge"]
        self.assertEqual(sha(EVIDENCE / reference["manifest_relative_path"]), reference["manifest_file_sha256"])
        claimed = definition.pop("definition_hash")
        canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), claimed)
        for relative, pairs in [
            ("metrics/analysis.metadata.json", [("case_metrics_path", "case_metrics_sha256"), ("statistics_path", "statistics_sha256")]),
            ("mrart_summary/summary.metadata.json", [("combined_metrics_path", "combined_metrics_sha256"), ("statistics_path", "statistics_sha256")]),
        ]:
            metadata = json.loads((EVIDENCE / relative).read_text())
            for path_key, hash_key in pairs:
                self.assertEqual(sha(EVIDENCE / metadata[path_key]), metadata[hash_key])


if __name__ == "__main__":
    unittest.main()
