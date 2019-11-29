from .core import ACSDataset, approximate_sum
import collections

__all__ = ["WorkerClass"]


class WorkerClass(ACSDataset):
    """
    Sex by Class of Worker for the Civilian Employed Population 16 Years and Over.
    """

    AGGREGATION = "count"
    UNIVERSE = "civilian employed population 16 years and over"
    TABLE_NAME = "B24080"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "universe",
            "002": "male_total",
            "003": "male_private_for_profit_wage_and_salary",
            "004": "male_employee_of_private_company",
            "005": "male_selfemployed_in_own_incorporated_business",
            "006": "male_private_not_for_profit_wage_and_salary",
            "007": "male_local_government",
            "008": "male_state_government",
            "009": "male_federal_government",
            "010": "male_selfemployed_in_own_not_incorporated_business",
            "011": "male_unpaid_family_workers",
            "012": "female_total",
            "013": "female_private_for_profit_wage_and_salary",
            "014": "female_employee_of_private_company",
            "015": "female_selfemployed_in_own_incorporated_business",
            "016": "female_private_not_for_profit_wage_and_salary",
            "017": "female_local_government",
            "018": "female_state_government",
            "019": "female_federal_government",
            "020": "female_selfemployed_in_own_not_incorporated_business",
            "021": "female_unpaid_family_workers",
        }
    )

    @classmethod
    def process(cls, df):

        groups = [
            "private_for_profit_wage_and_salary",
            "employee_of_private_company",
            "selfemployed_in_own_incorporated_business",
            "private_not_for_profit_wage_and_salary",
            "local_government",
            "state_government",
            "federal_government",
            "selfemployed_in_own_not_incorporated_business",
            "unpaid_family_workers",
        ]

        # Sum over male and female to add "total"
        for g in groups:

            # columns to sum
            cols_to_sum = [f"{tag}_{g}" for tag in ["male", "female"]]

            # do the aggregation
            newcols = [f"total_{g}", f"total_{g}_moe"]
            df[newcols] = df.apply(approximate_sum, cols=cols_to_sum, axis=1)

        return df

