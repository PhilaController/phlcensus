from ..core import Dataset, EPSG, data_dir
from ..regions import CensusTracts
import pandas as pd

__all__ = ["OpportunityAtlas"]


class OpportunityAtlas(Dataset):
    """
    Social mobility indicators from the Opportunity Atlas.

    Notes
    -----
    Two columns of interest:

    1. kfr_[race]_[gender]_p25: Mean household income rank for children whose
       parents were at the 25th percentile of the national income distribution.
       Incomes for children were measured as mean earnings in 2014-2015 when
       they were between the ages 31-37. Household income is defined as the sum
       of own and spouse’s income

    2. jail_[race]_[gender]_p25: Fraction of children born in 1978-1983 birth
       cohorts with parents at the 25th percentile of the national income
       distribution who were incarcerated on April 1st, 2010. Incarceration is
       defined as residing in a federal detention center, federal prison, state
       prison, local jail, residential correctional facility, military jail, or
       juvenile correctional facility

    Source
    ------
    https://www.census.gov/programs-surveys/ces/data/public-use-data/opportunity-atlas-data-tables.html
    """

    URL = "http://www2.census.gov/ces/opportunity/tract_outcomes_simple.csv"

    @classmethod
    def download(cls, **kwargs):

        # load the data and trim to philadelphia
        df = pd.read_csv(cls.URL).query("state == 42 and county==101")

        # Add geoid
        df["geo_id"] = df.apply(
            lambda row: f"{row['state']}{row['county']}{row['tract']:06d}", axis=1
        )

        # Only return pooled results for household income / incarceration rate
        cols = ["geo_id"]
        cols += [
            col
            for col in df.columns
            if (col.startswith("kfr_pooled") or col.startswith("jail_pooled"))
        ]
        cols += [
            col
            for col in df.columns
            if (col.startswith("pooled") and col.endswith("count"))
        ]
        df = df[cols]

        # add Census tracts
        tracts = CensusTracts.get().assign(geo_id=lambda df: df.geo_id.astype(str))

        # return merged with tracts
        return tracts.merge(df, on="geo_id")

