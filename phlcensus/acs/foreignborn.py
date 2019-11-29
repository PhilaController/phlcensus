from .core import ACSDataset

__all__ = ["ForeignBorn"]


class ForeignBorn(ACSDataset):
    """
    Place of birth by nativity and citizenship status.
    """

    AGGREGATION = "count"
    UNIVERSE = "total population"
    TABLE_NAME = "B05002"
    RAW_FIELDS = {"001": "universe", "002": "native", "013": "foreign_born"}

