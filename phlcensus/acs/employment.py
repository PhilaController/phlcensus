from .core import ACSDataset, approximate_ratio, approximate_sum
import collections

__all__ = ["EmploymentStatus", "EmploymentAge", "HoursWorked"]


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


class EmploymentAge(ACSDataset):
    """
    Employment by sex and age.
    """

    GROUPS = [
        "total_employed",
        "16_to_19_employed",
        "20_to_21_employed",
        "22_to_24_employed",
        "25_to_29_employed",
        "30_to_34_employed",
        "35_to_44_employed",
        "45_to_54_employed",
        "55_to_59_employed",
        "60_to_61_employed",
        "62_to_64_employed",
        "65_to_69_employed",
        "70_to_74_employed",
        "75_and_over_employed",
    ]
    UNIVERSE = "population 16 years and over"
    TABLE_NAME = "B23001"
    AGGREGATION = "count"

    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "007": "male_16_to_19_employed",
            "014": "male_20_to_21_employed",
            "021": "male_22_to_24_employed",
            "028": "male_25_to_29_employed",
            "035": "male_30_to_34_employed",
            "042": "male_35_to_44_employed",
            "049": "male_45_to_54_employed",
            "056": "male_55_to_59_employed",
            "063": "male_60_to_61_employed",
            "070": "male_62_to_64_employed",
            "075": "male_65_to_69_employed",
            "080": "male_70_to_74_employed",
            "085": "male_75_and_over_employed",
            "093": "female_16_to_19_employed",
            "100": "female_20_to_21_employed",
            "107": "female_22_to_24_employed",
            "114": "female_25_to_29_employed",
            "121": "female_30_to_34_employed",
            "128": "female_35_to_44_employed",
            "135": "female_45_to_54_employed",
            "142": "female_55_to_59_employed",
            "149": "female_60_to_61_employed",
            "156": "female_62_to_64_employed",
            "161": "female_65_to_69_employed",
            "166": "female_70_to_74_employed",
            "171": "female_75_and_over_employed",
        }
    )

    @classmethod
    def process(cls, df):
        """
        Create multiple sub-age groups.
        """

        # Calculate totals for both genders together
        for g in cls.GROUPS[1:]:

            # the columns to sum
            cols_to_sum = [f"{tag}_{g}" for tag in ["male", "female"]]

            # approximate the sum
            new_cols = [f"total_{g}", f"total_{g}_moe"]
            df[new_cols] = df.apply(approximate_sum, cols=cols_to_sum, axis=1)

        # Calculate custom group sets
        groupsets = collections.OrderedDict(
            {
                "16_to_21_employed": ["16_to_19_employed", "20_to_21_employed"],
                "22_to_29_employed": ["22_to_24_employed", "25_to_29_employed"],
                "30_to_44_employed": ["30_to_34_employed", "35_to_44_employed"],
                "45_to_64_employed": [
                    "45_to_54_employed",
                    "55_to_59_employed",
                    "60_to_61_employed",
                    "62_to_64_employed",
                ],
                "65_and_over_employed": [
                    "65_to_69_employed",
                    "70_to_74_employed",
                    "75_and_over_employed",
                ],
                "16_to_64_employed": [
                    "16_to_19_employed",
                    "20_to_21_employed",
                    "22_to_24_employed",
                    "25_to_29_employed",
                    "30_to_34_employed",
                    "35_to_44_employed",
                    "45_to_54_employed",
                    "55_to_59_employed",
                    "60_to_61_employed",
                    "62_to_64_employed",
                ],
            }
        )

        # Sum over the custom groups
        for groupset, group_list in groupsets.items():
            for tag in ["total", "male", "female"]:

                # cols to sum over
                cols_to_sum = [f"{tag}_{f}" for f in group_list]

                # do the aggregation
                newcols = [f"{tag}_{groupset}", f"{tag}_{groupset}_moe"]
                df[newcols] = df.apply(approximate_sum, cols=cols_to_sum, axis=1)

        return df


class HoursWorked(ACSDataset):
    """
    Aggregate usual hours worked by population 16 to 64 Years who 
    have worked in the past 12 months.
    """

    UNIVERSE = "population 16 years to 64"
    TABLE_NAME = "B23018"
    AGGREGATION = "count"

    RAW_FIELDS = collections.OrderedDict(
        {"001": "total_hours", "002": "male_hours", "003": "female_hours"}
    )

