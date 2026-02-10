import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from dataloader import DATASET_CONFIGS


class DatasetConfigsTests(unittest.TestCase):
    def test_mimiciv_cm_versioned_keys_exist(self) -> None:
        expected = {f"mimiciv-cm-{version}" for version in ["3.0", "3.1", "3.2", "3.3", "3.4"]}
        self.assertTrue(expected.issubset(set(DATASET_CONFIGS)))

    def test_representative_mimiciv_config_values_unchanged(self) -> None:
        mimiciv_cfg = DATASET_CONFIGS["mimic-iv"]
        self.assertEqual(mimiciv_cfg["identifier"], "mimic-iv")
        self.assertEqual(mimiciv_cfg["split"], "test")
        self.assertEqual(mimiciv_cfg["subsets"], ["icd10"])
        self.assertEqual(
            mimiciv_cfg["options"],
            {"subset_size": 300, "adapter": "MimicIdentifyAdapter"},
        )

    def test_representative_versioned_mimiciv_cm_values_unchanged(self) -> None:
        versioned_cfg = DATASET_CONFIGS["mimiciv-cm-3.0"]
        self.assertEqual(versioned_cfg["identifier"], "mimiciv-cm-3.0")
        self.assertEqual(versioned_cfg["split"], "test")
        self.assertEqual(versioned_cfg["subsets"], ["icd10"])
        self.assertEqual(
            versioned_cfg["options"],
            {"subset_size": 1000, "adapter": "MimicIdentifyAdapter"},
        )


if __name__ == "__main__":
    unittest.main()
