from .core import ACSDataset
import collections

__all__ = ["SchoolEnrollment"]


class SchoolEnrollment(ACSDataset):
    """ 
    School enrollment
    """

    AGGREGATION = "count"
    UNIVERSE = "population 3 years and older enrolled in school"
    TABLE_NAME = "S1401_C01"
    RAW_FIELDS = collections.OrderedDict(
        {
            "001": "total_3_and_over",
            "002": "nursery_preschool",
            "004": "kindergarten",
            "005": "grades_1_to_4",
            "006": "grades_5_to_8",
            "007": "grades_9_to_12",
            "008": "college_undergrad",
            "009": "graduate_professional",
            "013": "total_3_to_4",
            "014": "enrolled_3_to_4",
            "015": "total_5_to_9",
            "016": "enrolled_5_to_9",
            "017": "total_10_to_14",
            "018": "enrolled_10_to_14",
            "019": "total_15_to_17",
            "020": "enrolled_15_to_17",
            "021": "total_18_to_19",
            "022": "enrolled_18_to_19",
            "023": "total_20_to_24",
            "024": "enrolled_20_to_24",
            "025": "total_25_to_34",
            "026": "enrolled_25_to_34",
            "027": "total_35_and_over",
            "028": "enrolled_35_and_over",
        }
    )

