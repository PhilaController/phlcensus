from .core import ACSDataset
from .. import agg
import collections

__all__ = ["EmploymentStatus"]


class EmploymentStatus(ACSDataset):
    """
    Employment status for the population 16 years and older.
    """

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

        # Unemployment rate
        newcol = "unemployment_rate"
        df[[newcol, f"{newcol}_moe"]] = df.apply(
            agg.approximate_ratio,
            cols=["civilian_unemployed", "in_labor_force"],
            axis=1,
        )

        return df

