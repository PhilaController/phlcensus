from phlcensus.acs import DATASETS
from phlcensus.regions import *
import pytest


@pytest.mark.parametrize("cls", DATASETS.values())
def test_tract_level(cls):

    data = cls.get(level="tract")
    tracts = CensusTracts.get()
    assert len(data) == len(tracts)


@pytest.mark.parametrize("cls", DATASETS.values())
def test_city_level(cls):

    data = cls.get(level="city")
    assert len(data) == 1


@pytest.mark.parametrize("cls", DATASETS.values())
def test_puma_level(cls):
    data = cls.get(level="puma")
    pumas = PUMAs.get()
    assert len(data) == len(pumas)


@pytest.mark.parametrize("cls", DATASETS.values())
def test_count_aggregation(cls):

    if cls.AGGREGATION == "count":

        # data by census tract
        data_by_tract = cls.get(level="tract")

        # columns to test
        cols = list(cls.RAW_FIELDS.values())

        # NTAs
        hoods = NTAs.get()

        # Test other levels
        for level in ["nta"]:
            data = cls.get(level=level)

            # check the number of hoods
            assert len(data) == len(hoods)

            # check the total sum
            assert (data_by_tract[cols].sum() == data[cols].sum()).all()
