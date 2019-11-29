from .core import ACSDataset
import collections

__all__ = ["PovertyStatus"]


class PovertyStatus(ACSDataset):
    """ 
    Poverty status by gender.
    """

    AGGREGATION = "count"
    UNIVERSE = "population for whom poverty status is determined"
    TABLE_NAME = "B17001"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "total_below_poverty_level",
            "003": "male_below_poverty_level",
            "017": "female_below_poverty_level",
        }
    )

