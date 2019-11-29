from .core import ACSDataset, DEFAULT_YEAR
import collections
import numpy as np

__all__ = ["MedianAge"]


class MedianAge(ACSDataset):
    """
    Median age by sex.
    """

    AGGREGATION = "median"
    UNIVERSE = "total population"
    TABLE_NAME = "B01002"
    RAW_FIELDS = collections.OrderedDict(
        {"001": "median", "002": "male", "003": "female"}
    )

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
        from .age import Age

        # the underlying data
        data = Age.get(level="tract", year=year, N=N)

        # map median cols to dist cols
        cols = [("median", "total"), ("male", "male"), ("female", "female")]

        # yield the right subsets
        for (out_col, prefix) in cols:

            # the aggregation bins
            bins = Age._get_aggregation_bins(prefix=prefix)

            cols = [b[-1] for b in bins]
            cols += [f"{col}_moe" for col in cols]
            yield data[["geometry", "geo_id", "geo_name"] + cols], bins, out_col

