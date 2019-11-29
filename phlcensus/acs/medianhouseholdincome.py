from .core import ACSDataset, DEFAULT_YEAR
import numpy as np

__all__ = ["MedianHouseholdIncome"]


class MedianHouseholdIncome(ACSDataset):
    """
    Median household income in the past 12 months (in inflation-adjusted dollars).
    """

    AGGREGATION = "median"
    UNIVERSE = "households"
    TABLE_NAME = "B19013"
    RAW_FIELDS = {"001": "median"}

    @classmethod
    def process(cls, df):

        # Set values < 0 to null
        cols = list(cls.RAW_FIELDS.values())
        for col in cols:
            invalid = df[col] < 0
            df.loc[invalid, [col, f"{col}_moe"]] = np.nan

        return df

    @staticmethod
    def _median_aggregator(year=DEFAULT_YEAR, N=5):
        """
        Function that helps calculate the median value from 
        the underlying distribution of raw counts.

        Yields
        ------
        data : 
            the raw count data in bins
        bins : list
            the matching distribution bins, as returned by the 
            `get_aggregation_bins()` function
        out_col : 
            string specifying name of the calculated column
        """
        from .householdincome import HouseholdIncome

        # the underlying data
        data = HouseholdIncome.get(level="tract", year=year, N=N)

        # the aggregation bins
        bins = HouseholdIncome._get_aggregation_bins()

        cols = [b[-1] for b in bins]
        cols += [f"{col}_moe" for col in cols]
        yield data[["geometry", "geo_id", "geo_name"] + cols], bins, "median"

