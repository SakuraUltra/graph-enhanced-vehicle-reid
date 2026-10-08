"""Regression coverage for the medium/large silently-using-small failure."""

from pathlib import Path
import tempfile
import unittest

from utils.vehicleid_splits import resolve_vehicleid_test_list


class VehicleIDSplitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.splits = self.root / "train_test_split"
        self.splits.mkdir()
        # Different identity sets let the test detect accidental small-set reuse.
        (self.splits / "test_list_800.txt").write_text("img1 10\n")
        (self.splits / "test_list_1600.txt").write_text("img2 20\nimg3 21\n")
        (self.splits / "test_list_2400.txt").write_text("img4 30\nimg5 31\nimg6 32\n")

    def identities_for(self, size):
        filename = resolve_vehicleid_test_list(self.root, size)
        return [int(line.split()[1]) for line in (self.splits / filename).read_text().splitlines()]

    def test_small_selects_800_file(self):
        self.assertEqual(self.identities_for("small"), [10])

    def test_medium_selects_1600_file(self):
        self.assertEqual(self.identities_for("medium"), [20, 21])

    def test_large_selects_2400_file(self):
        self.assertEqual(self.identities_for("large"), [30, 31, 32])

    def test_missing_medium_does_not_fall_back_to_existing_small(self):
        (self.splits / "test_list_1600.txt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "test_list_1600.txt"):
            resolve_vehicleid_test_list(self.root, "medium")

    def test_missing_large_does_not_fall_back_to_existing_small(self):
        (self.splits / "test_list_2400.txt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "test_list_2400.txt"):
            resolve_vehicleid_test_list(self.root, "large")

    def test_missing_small_is_reported(self):
        (self.splits / "test_list_800.txt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "test_list_800.txt"):
            resolve_vehicleid_test_list(str(self.root), "small")

    def test_directory_is_not_a_split_file(self):
        (self.splits / "test_list_1600.txt").unlink()
        (self.splits / "test_list_1600.txt").mkdir()
        with self.assertRaises(FileNotFoundError):
            resolve_vehicleid_test_list(self.root, "medium")

    def test_unknown_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown VehicleID test size"):
            resolve_vehicleid_test_list(self.root, "extra-large")


if __name__ == "__main__":
    unittest.main()
