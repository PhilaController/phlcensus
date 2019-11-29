from .core import ACSDataset, approximate_sum
import collections

__all__ = ["EducationalAttainment"]


class EducationalAttainment(ACSDataset):
    """
    Sex by educational attainment for the population 25 years and over.
    """

    AGGREGATION = "count"
    UNIVERSE = "population 25 years and over"
    TABLE_NAME = "B15002"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "male_total",
            "003": "male_no_schooling",
            "004": "male_nursery_to_4th_grade",
            "005": "male_5th_and_6th_grade",
            "006": "male_7th_and_8th_grade",
            "007": "male_9th_grade",
            "008": "male_10th_grade",
            "009": "male_11th_grade",
            "010": "male_12th_grade_no_diploma",
            "011": "male_high_school_graduate",
            "012": "male_less_than_1_year_college",
            "013": "male_1_or_more_years_college",
            "014": "male_associates_degree",
            "015": "male_bachelors_degree",
            "016": "male_masters_degree",
            "017": "male_professional_school_degree",
            "018": "male_doctorate_degree",
            "019": "female_total",
            "020": "female_no_schooling",
            "021": "female_nursery_to_4th_grade",
            "022": "female_5th_and_6th_grade",
            "023": "female_7th_and_8th_grade",
            "024": "female_9th_grade",
            "025": "female_10th_grade",
            "026": "female_11th_grade",
            "027": "female_12th_grade_no_diploma",
            "028": "female_high_school_graduate",
            "029": "female_less_than_1_year_college",
            "030": "female_1_or_more_years_college",
            "031": "female_associates_degree",
            "032": "female_bachelors_degree",
            "033": "female_masters_degree",
            "034": "female_professional_school_degree",
            "035": "female_doctorate_degree",
        }
    )

    @classmethod
    def process(cls, df):

        # Calculate totals for both genders together
        groups = [
            "no_schooling",
            "nursery_to_4th_grade",
            "5th_and_6th_grade",
            "7th_and_8th_grade",
            "9th_grade",
            "10th_grade",
            "11th_grade",
            "12th_grade_no_diploma",
            "high_school_graduate",
            "less_than_1_year_college",
            "1_or_more_years_college",
            "associates_degree",
            "bachelors_degree",
            "masters_degree",
            "professional_school_degree",
            "doctorate_degree",
        ]

        # Sum over gender
        for g in groups:

            # sum over these columns
            cols_to_sum = [f"{tag}_{g}" for tag in ["male", "female"]]

            # do the aggregation
            newcols = [f"total_{g}", f"total_{g}_moe"]
            df[newcols] = df.apply(approximate_sum, cols=cols_to_sum, axis=1)

        # Calculate our custom groups
        groupsets = collections.OrderedDict(
            {
                "less_than_high_school": [
                    "no_schooling",
                    "nursery_to_4th_grade",
                    "5th_and_6th_grade",
                    "7th_and_8th_grade",
                    "9th_grade",
                    "10th_grade",
                    "11th_grade",
                    "12th_grade_no_diploma",
                ],
                "some_college": ["less_than_1_year_college", "1_or_more_years_college"],
                "masters_or_higher": [
                    "masters_degree",
                    "professional_school_degree",
                    "doctorate_degree",
                ],
                "bachelors_or_higher": [
                    "bachelors_degree",
                    "masters_degree",
                    "professional_school_degree",
                    "doctorate_degree",
                ],
            }
        )
        for groupset, group_list in groupsets.items():
            for tag in ["total", "male", "female"]:

                # columns to sum
                cols_to_sum = [f"{tag}_{f}" for f in group_list]

                # new columns
                newcols = [f"{tag}_{groupset}", f"{tag}_{groupset}_moe"]
                df[newcols] = df.apply(approximate_sum, cols=cols_to_sum, axis=1)

        return df
