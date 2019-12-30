from .core import ACSDataset
import collections

__all__ = ["SNAP"]


class SNAP(ACSDataset):
    """
    Household received Food Stamps/SNAP in the past 12 months
    """

    UNIVERSE = "household"
    TABLE_NAME = "B22003"
    AGGREGATION = "count"
    RAW_FIELDS = collections.OrderedDict(
        {"001": "universe", "002": "recipient", "005": "nonrecipient"}
    )
