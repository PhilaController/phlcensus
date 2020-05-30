from .core import ACSDataset
import collections

__all__ = ["Internet"]


class Internet(ACSDataset):
    """
    PRESENCE AND TYPES OF INTERNET SUBSCRIPTIONS IN HOUSEHOLD
    """

    AGGREGATION = "count"
    UNIVERSE = "households"
    TABLE_NAME = "B28002"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "internet_any_source",
            "003": "dialup_only",
            "004": "broadband_any_source",
            "005": "cellular_data",
            "006": "cellular_data_only",
            "007": "broadband_cable_fiber_or_dsl",
            "008": "broadband_only",
            "009": "satellite",
            "010": "satellite_only",
            "011": "other_only",
            "012": "internet_without_subscription",
            "013": "no_internet",
        }
    )
