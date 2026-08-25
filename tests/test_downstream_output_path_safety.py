from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DownstreamOutputPathSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lesion = load_module(
            "build_atlas_lesion_size_failure_tables",
            ROOT
            / "scripts/supporting_experiments/atlas_lesion_size_stratification/scripts/"
            / "build_atlas_lesion_size_failure_tables.py",
        )
        cls.stats = load_module(
            "build_paper_statistical_inference",
            ROOT
            / "scripts/supporting_experiments/statistical_inference/scripts/"
            / "build_paper_statistical_inference.py",
        )
        cls.helpers = [cls.lesion.check_output_paths_safe, cls.stats.check_output_paths_safe]

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / ".gitignore").write_text("ignored-output/\n", encoding="utf-8")
        protected = self.repo / "protected.txt"
        protected.write_text("source\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", ".gitignore", "protected.txt"], check=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ignored_output_inside_repo_is_accepted(self) -> None:
        for helper in self.helpers:
            helper([self.repo / "ignored-output" / "new" / "result.csv"], repo_root=self.repo)

    def test_tracked_or_nonignored_output_inside_repo_is_rejected(self) -> None:
        for helper in self.helpers:
            with self.subTest(helper=helper.__module__, kind="tracked"):
                with self.assertRaisesRegex(RuntimeError, "inside the repository"):
                    helper([self.repo / "protected.txt"], repo_root=self.repo)
            with self.subTest(helper=helper.__module__, kind="nonignored"):
                with self.assertRaisesRegex(RuntimeError, "inside the repository"):
                    helper([self.repo / "source-output" / "result.csv"], repo_root=self.repo)

    def test_existing_external_output_directory_is_accepted(self) -> None:
        external = self.root / "external-existing"
        external.mkdir()
        for helper in self.helpers:
            helper([external], repo_root=self.repo)

    def test_nonexistent_external_output_is_accepted_with_valid_parent(self) -> None:
        external_root = self.root / "external-root"
        external_root.mkdir()
        external_alias = self.root / "external-alias"
        external_alias.symlink_to(external_root, target_is_directory=True)
        external = external_alias / "new-run" / "result.csv"
        for helper in self.helpers:
            helper([external], repo_root=self.repo)

    def test_existing_output_overwrite_protection_is_unchanged(self) -> None:
        existing_file = self.root / "existing.csv"
        existing_file.write_text("existing\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.lesion.ensure_new_path(existing_file)

        existing_dir = self.root / "existing-output"
        existing_dir.mkdir()
        args = argparse.Namespace(output_dir=existing_dir, output_root=self.root)
        with self.assertRaises(FileExistsError):
            self.stats.choose_output_dir(args)


if __name__ == "__main__":
    unittest.main()
