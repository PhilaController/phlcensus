from .core import ACSDataset
import collections

__all__ = ["MedianEarnings"]


class MedianEarnings(ACSDataset):
    """
    Median earnings by sex.
    """

    UNIVERSE = "population 16 years and over with earnings"
    TABLE_NAME = "B20002"
    RAW_FIELDS = collections.OrderedDict(
        {"001": "total", "002": "male", "003": "female"}
    )

