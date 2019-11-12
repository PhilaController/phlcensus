from .core import ACSDataset
import collections
import numpy as np

__all__ = ["Earnings"]


class Earnings(ACSDataset):
    """
    Earnings by sex.
    """

    GROUPS = [
        "total",
        "1_to_2499",
        "2500_to_4999",
        "5000_to_7499",
        "7500_to_9999",
        "10000_to_12499",
        "12500_to_14999",
        "15000_to_17499",
        "17500_to_19999",
        "20000_to_22499",
        "22500_to_24999",
        "25000_to_29999",
        "30000_to_34999",
        "35000_to_39999",
        "40000_to_44999",
        "45000_to_49999",
        "50000_to_54999",
        "55000_to_64999",
        "65000_to_74999",
        "75000_to_99999",
        "100000_or_more",
    ]
    UNIVERSE = "population 16 years and over with earnings"
    TABLE_NAME = "B20001"
    RAW_FIELDS = collections.OrderedDict({"001": "universe"})

    cnt = 2
    for prefix in ["male", "female"]:
        for g in GROUPS:
            RAW_FIELDS[f"{cnt:03d}"] = f"{prefix}_{g}"
            cnt += 1

    @classmethod
    def get_aggregation_bins(cls, prefix="total"):
        """
        Return the aggregation bins for calculating the median
        earnings from the distribution. 

        Returns a list of the form (start, stop, column_name):
        """
        if prefix not in ["total", "male", "female"]:
            raise ValueError("allowed prefix values are 'total', 'male', and 'female'")

        bins = []
        for i, g in enumerate(cls.GROUPS[1:]):
            if i == 0:
                start, end = 0, 4
            elif g.endswith("or_more"):
                start = float(g.split("_")[0])
                end = np.inf
            else:
                fields = g.split("_to_")
                if len(fields) == 2:
                    start, end = map(float, fields)
                else:
                    start = end = float(fields[0])

            bins.append((start, end, f"{prefix}_{g}"))

        return bins
