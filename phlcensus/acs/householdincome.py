from .core import ACSDataset
import collections
import numpy as np

__all__ = ["HouseholdIncome"]


class HouseholdIncome(ACSDataset):
    """
    Household income in the past 12 months (in inflation-adjusted dollars).
    """

    AGGREGATION = "count"
    UNIVERSE = "households"
    TABLE_NAME = "B19001"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "0_to_9999",
            "003": "10000_to_14999",
            "004": "15000_to_19999",
            "005": "20000_to_24999",
            "006": "25000_to_29999",
            "007": "30000_to_34999",
            "008": "35000_to_39999",
            "009": "40000_to_44999",
            "010": "45000_to_49999",
            "011": "50000_to_59999",
            "012": "60000_to_74999",
            "013": "75000_to_99999",
            "014": "100000_to_124999",
            "015": "125000_to_149999",
            "016": "150000_to_199999",
            "017": "200000_or_more",
        }
    )

    @classmethod
    def _get_aggregation_bins(cls):
        """
        Return the aggregation bins for calculating the median
        household income from the distribution. 

        Returns
        -------
        bins : list of tuples
            tuples of (start, stop, column name)
        """

        bins = []
        for i in range(3, 18):
            column = cls.RAW_FIELDS[f"{i:03d}"]
            if column.endswith("or_more"):
                start = float(column.split("_")[0])
                end = np.inf
            else:
                start, end = map(float, column.split("_to_"))

            bins.append((start, end, column))

        return bins

