from phlcensus.external import *
import pytest


@pytest.mark.parametrize("cls", [OpportunityAtlas])
def test_get(cls):
    data = cls.get()

