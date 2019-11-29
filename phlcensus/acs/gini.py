from .core import ACSDataset

__all__ = ["GiniIncomeInequality"]


class GiniIncomeInequality(ACSDataset):
    """
    Gini index of income inequality.
    """

    AGGREGATION = None  # dont aggregate this one
    UNIVERSE = "households"
    TABLE_NAME = "B19083"
    RAW_FIELDS = {"001": "gini_index"}

