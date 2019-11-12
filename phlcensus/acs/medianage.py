from .core import ACSDataset
import collections

__all__ = ["MedianAge"]


class MedianAge(ACSDataset):
    """
    Median age by sex.
    """

    UNIVERSE = "total population"
    TABLE_NAME = "B01002"
    RAW_FIELDS = collections.OrderedDict(
        {"001": "median", "002": "male", "003": "female"}
    )
