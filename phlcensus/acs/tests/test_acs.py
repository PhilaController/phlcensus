from phlcensus.acs import DATASETS
import pytest


@pytest.mark.parametrize("cls", DATASETS.values())
def test_tract_level(cls):

    data = cls.get(level="tract")


@pytest.mark.parametrize("cls", DATASETS.values())
def test_city_level(cls):

    data = cls.get(level="city")
    assert len(data) == 1


@pytest.mark.parametrize("cls", DATASETS.values())
def test_puma_level(cls):

    data = cls.get(level="puma")
