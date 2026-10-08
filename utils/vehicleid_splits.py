"""Resolve standard VehicleID evaluation splits without silently changing size."""

from pathlib import Path


VEHICLEID_TEST_LISTS = {
    "small": "test_list_800.txt",
    "medium": "test_list_1600.txt",
    "large": "test_list_2400.txt",
}


def resolve_vehicleid_test_list(dataset_root, test_size):
    """Return the requested split filename, requiring that exact file to exist."""
    if test_size not in VEHICLEID_TEST_LISTS:
        choices = ", ".join(VEHICLEID_TEST_LISTS)
        raise ValueError(f"Unknown VehicleID test size {test_size!r}; choose {choices}")
    filename = VEHICLEID_TEST_LISTS[test_size]
    path = Path(dataset_root) / "train_test_split" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"VehicleID {test_size} split not found: {path}. "
            "Provide the requested split; evaluation will not fall back to another size."
        )
    return filename
