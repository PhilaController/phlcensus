from .core import ACSDataset
import collections
import numpy as np

__all__ = [
    "HousingValue",
    "MedianHousingValue",
    "GrossRent",
    "MedianGrossRent",
    "Tenure",
]


class HousingValue(ACSDataset):
    """
    Housing value.
    """

    UNIVERSE = "owner-occupied housing units"
    TABLE_NAME = "B25075"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "0_to_10000",
            "003": "10000_to_14999",
            "004": "15000_to_19999",
            "005": "20000_to_24999",
            "006": "25000_to_29999",
            "007": "30000_to_34999",
            "008": "35000_to_39999",
            "009": "40000_to_49999",
            "010": "50000_to_59999",
            "011": "60000_to_69999",
            "012": "70000_to_79999",
            "013": "80000_to_89999",
            "014": "90000_to_99999",
            "015": "100000_to_124999",
            "016": "125000_to_149999",
            "017": "150000_to_174999",
            "018": "175000_to_199999",
            "019": "200000_to_249999",
            "020": "250000_to_299999",
            "021": "300000_to_399999",
            "022": "400000_to_499999",
            "023": "500000_to_749999",
            "024": "750000_to_999999",
            "025": "1000000_to_1499999",
            "026": "1500000_to_1999999",
            "027": "2000000_or_more",
        }
    )

    @classmethod
    def get_aggregation_bins(cls):
        """
        """

        bins = []
        for i in range(3, 28):
            column = cls.RAW_FIELDS[f"{i:03d}"]
            if column.endswith("or_more"):
                start = float(column.split("_")[0])
                end = np.inf
            else:
                start, end = map(float, column.split("_to_"))

            bins.append((start, end, column))

        return bins


class MedianHousingValue(ACSDataset):
    """
    Median housing value.
    """

    UNIVERSE = "owner-occupied housing units"
    TABLE_NAME = "B25077"
    RAW_FIELDS = {"001": "median"}


class GrossRent(ACSDataset):
    """
    Gross rent (paying cash).
    """

    UNIVERSE = "renter-occupied housing units paying cash rent"
    TABLE_NAME = "B25063"
    RAW_FIELDS = collections.OrderedDict(
        {
            "002": "universe",
            "003": "0_to_99",
            "004": "100_to_149",
            "005": "150_to_199",
            "006": "200_to_249",
            "007": "250_to_299",
            "008": "300_to_349",
            "009": "350_to_399",
            "010": "400_to_449",
            "011": "450_to_499",
            "012": "500_to_549",
            "013": "550_to_599",
            "014": "600_to_649",
            "015": "650_to_699",
            "016": "700_to_749",
            "017": "750_to_799",
            "018": "800_to_899",
            "019": "900_to_999",
            "020": "1000_to_1249",
            "021": "1250_to_1499",
            "022": "1500_to_1999",
            "023": "2000_to_2499",
            "024": "2500_to_2999",
            "025": "3000_to_3499",
            "026": "3500_or_more",
        }
    )

    @classmethod
    def get_aggregation_bins(cls):
        """
        """

        bins = []
        for i in range(3, 27):
            column = cls.RAW_FIELDS[f"{i:03d}"]
            if column.endswith("or_more"):
                start = float(column.split("_")[0])
                end = np.inf
            else:
                start, end = map(float, column.split("_to_"))

            bins.append((start, end, column))

        return bins


class MedianGrossRent(ACSDataset):
    """
    Median gross rent (dollars).
    """

    UNIVERSE = "renter-occupied housing units paying cash rent"
    TABLE_NAME = "B25064"
    RAW_FIELDS = {"001": "median"}


class Tenure(ACSDataset):
    """
    Occupied housing units. Owner or renter.
    """

    UNIVERSE = "occupied housing units"
    TABLE_NAME = "B25003"
    RAW_FIELDS = collections.OrderedDict(
        {"001": "universe", "002": "owner_occupied", "003": "renter_occupied"}
    )

