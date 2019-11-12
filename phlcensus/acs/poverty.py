from .core import ACSDataset
import collections

__all__ = ["PovertyStatus", "PovertyBySex"]


class PovertyStatus(ACSDataset):
    """
    Poverty in terms of 12 month income below/above poverty level.
    """

    UNIVERSE = "population for whom poverty status is determined"
    TABLE_NAME = "B17001"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "income_past12months_below_poverty_level",
            "031": "income_past12months_at_or_above_poverty_level",
        }
    )


class PovertyBySex(ACSDataset):
    """ 
    Poverty status by gender.
    """

    UNIVERSE = "population for whom poverty status is determined"
    TABLE_NAME = "B17001"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "total",
            "002": "total_below_poverty_level",
            "003": "male_below_poverty_level",
            "017": "female_below_poverty_level",
        }
    )

