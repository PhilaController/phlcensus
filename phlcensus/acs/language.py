from .core import ACSDataset, approximate_sum
import collections

__all__ = ["HouseholdLanguage"]


class HouseholdLanguage(ACSDataset):
    """
    Household language by household limited English speaking status.
    """

    AGGREGATION = "count"
    UNIVERSE = "households"
    TABLE_NAME = "C16002"
    YEARS = [2017, 2016]
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "only_english",
            "003": "total_spanish",
            "004": "spanish_limited_english",
            "005": "spanish_not_limited_english",
            "006": "total_other_indo_european",
            "007": "other_indo_european_limited_english",
            "008": "other_indo_european_not_limited_english",
            "009": "total_asian_and_pacific_islander",
            "010": "asian_and_pacific_islander_limited_english",
            "011": "asian_and_pacific_islander_not_limited_english",
            "012": "total_other",
            "013": "other_limited_english",
            "014": "other_not_limited_english",
        }
    )

    @classmethod
    def process(cls, df):

        # add a more general other category
        languages = [
            "spanish",
            "other_indo_european",
            "asian_and_pacific_islander",
            "other",
        ]

        # not limited English
        newcol = "total_not_limited_english"
        newcols = [newcol, f"{newcol}_moe"]
        cols_to_sum = ["only_english"] + [f"{l}_not_limited_english" for l in languages]
        df[newcols] = df.apply(approximate_sum, cols=cols_to_sum, axis=1)

        # limited English
        newcol = "total_limited_english"
        newcols = [newcol, f"{newcol}_moe"]
        cols_to_sum = [f"{l}_limited_english" for l in languages]
        df[newcols] = df.apply(approximate_sum, cols=cols_to_sum, axis=1)

        return df

