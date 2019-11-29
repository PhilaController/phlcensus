from .core import ACSDataset, approximate_ratio
import collections

__all__ = ["EmploymentStatus"]


class EmploymentStatus(ACSDataset):
    """
    Employment status for the population 16 years and older.
    """

    AGGREGATION = "count"
    UNIVERSE = "population 16 years and over"
    TABLE_NAME = "B23025"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "in_labor_force",
            "003": "civilian",
            "004": "civilian_employed",
            "005": "civilian_unemployed",
            "006": "armed_forces",
            "007": "not_in_labor_force",
        }
    )

    @classmethod
    def process(cls, df):

        # cols to ratio
        cols_to_ratio = ["civilian_unemployed", "in_labor_force"]

        # Unemployment rate
        newcols = ["unemployment_rate", "unemployment_rate_moe"]
        df[newcols] = df.apply(approximate_ratio, cols=cols_to_ratio, axis=1)

        return df

