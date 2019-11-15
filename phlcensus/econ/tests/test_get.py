from phlcensus.econ import *
import pytest


@pytest.mark.parametrize("cls", [JobsByHomeTract, JobsByWorkTract])
def test_get(cls):
    data = cls.get()

