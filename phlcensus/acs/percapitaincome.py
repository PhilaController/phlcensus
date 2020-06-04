from .core import ACSDataset
import collections

__all__ = ["PerCapitaIncome"]


class PerCapitaIncome(ACSDataset):
    """ 
    PER CAPITA INCOME IN THE PAST 12 MONTHS (IN 2018 INFLATION-ADJUSTED DOLLARS) 
    """

    AGGREGATION = None
    UNIVERSE = "total population"
    TABLE_NAME = "B19301"
    RAW_FIELDS = collections.OrderedDict({"001": "per_capita_income"})
