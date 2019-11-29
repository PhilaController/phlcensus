from .core import ACSDataset
import collections

__all__ = ["Mobility"]


class Mobility(ACSDataset):
    """
    Geographical mobility in the past year by sex for current residence in the US.
    """

    AGGREGATION = "count"
    UNIVERSE = "population 1 year and over"
    TABLE_NAME = "B07003"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "004": "same_house",
            "007": "moved_within_county",
            "010": "moved_from_different_county_in_same_state",
            "013": "moved_from_different_state",
            "016": "moved_from_abroad",
        }
    )
