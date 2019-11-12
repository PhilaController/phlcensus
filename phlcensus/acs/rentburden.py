from .core import ACSDataset
from .. import agg
import collections

__all__ = ["RentBurden"]


class RentBurden(ACSDataset):
    """
    Gross rent as a percentage of household income in the past 12 months.
    """

    UNIVERSE = "renter-occupied housing units"
    TABLE_NAME = "B25070"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "less_than_10",
            "003": "10_to_15",
            "004": "15_to_20",
            "005": "20_to_25",
            "006": "25_to_30",
            "007": "30_to_35",
            "008": "35_to_40",
            "009": "40_to_50",
            "010": "more_than_50",
        }
    )

    @classmethod
    def process(cls, df):

        # More than 35% is defined as rent burdened
        newcol = "more_than_35"
        df[[newcol, f"{newcol}_moe"]] = df.apply(
            agg.approximate_sum, cols=["35_to_40", "40_to_50", "more_than_50"], axis=1
        )

        # As a percent
        newcol = "percent_more_than_35"
        df[[newcol, f"{newcol}_moe"]] = df.apply(
            agg.approximate_ratio, cols=["more_than_35", "universe"], axis=1
        )

        return df

