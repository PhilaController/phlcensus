from phlcensus.economic import *
import pytest


@pytest.mark.parametrize("cls", [SummaryLODES, DetailedLODES])
def test_lodes_work(cls):

    # Load the data by census tract
    data_by_tract = cls.get(level="tract", kind="work")

    # columns to compare
    cols = [
        col
        for col in data_by_tract
        if not col.startswith("geo") and col != "total_work_at_home"
    ]

    # Test other levels
    for level in ["nta", "puma"]:
        data = cls.get(level=level, kind="work")
        assert (data_by_tract[cols].sum() == data[cols].sum()).all()


@pytest.mark.parametrize("cls", [SummaryLODES, DetailedLODES])
def test_lodes_home(cls):

    # Load the data by census tract
    data_by_tract = cls.get(level="tract", kind="home")

    # columns to compare
    cols = [
        col
        for col in data_by_tract
        if not col.startswith("geo") and col != "total_work_at_home"
    ]

    # Test other levels
    for level in ["nta", "puma"]:
        data = cls.get(level=level, kind="home")
        assert (data_by_tract[cols].sum() == data[cols].sum()).all()
