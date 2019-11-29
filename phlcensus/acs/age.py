from .core import ACSDataset, approximate_sum
import collections
import numpy as np

__all__ = ["Age"]


class Age(ACSDataset):
    """
    Population by sex and age.
    """

    GROUPS = [
        "total",
        "under_5",
        "5_to_9",
        "10_to_14",
        "15_to_17",
        "18_to_19",
        "20",
        "21",
        "22_to_24",
        "25_to_29",
        "30_to_34",
        "35_to_39",
        "40_to_44",
        "45_to_49",
        "50_to_54",
        "55_to_59",
        "60_to_61",
        "62_to_64",
        "65_to_66",
        "67_to_69",
        "70_to_74",
        "75_to_79",
        "80_to_84",
        "85_and_over",
    ]

    AGGREGATION = "count"
    UNIVERSE = "total population"
    TABLE_NAME = "B01001"
    RAW_FIELDS = collections.OrderedDict({"001": "universe"})

    cnt = 2
    for prefix in ["male", "female"]:
        for g in GROUPS:
            RAW_FIELDS[f"{cnt:03d}"] = f"{prefix}_{g}"
            cnt += 1

    @classmethod
    def process(cls, df):
        """
        Create multiple sub-age groups and generational groups.
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
                "0_to_17": ["under_5", "5_to_9", "10_to_14", "15_to_17"],
                "5_to_17": ["5_to_9", "10_to_14", "15_to_17"],
                "0_9": ["under_5", "5_to_9"],
                "10_to_17": ["10_to_14", "15_to_17"],
                "18_to_34": [
                    "18_to_19",
                    "20",
                    "21",
                    "22_to_24",
                    "25_to_29",
                    "30_to_34",
                ],
                "35_to_49": ["35_to_39", "40_to_44", "45_to_49"],
                "50_to_64": ["50_to_54", "55_to_59", "60_to_61", "62_to_64"],
                "65_and_over": [
                    "65_to_66",
                    "67_to_69",
                    "70_to_74",
                    "75_to_79",
                    "80_to_84",
                    "85_and_over",
                ],
                "silent": ["75_to_79", "80_to_84", "85_and_over"],
                "boomers": [
                    "55_to_59",
                    "60_to_61",
                    "62_to_64",
                    "65_to_66",
                    "67_to_69",
                    "70_to_74",
                ],
                "gen_x": ["40_to_44", "45_to_49", "50_to_54"],
                "millennials": ["22_to_24", "25_to_29", "30_to_34", "35_to_39"],
                "gen_z": [
                    "under_5",
                    "5_to_9",
                    "10_to_14",
                    "15_to_17",
                    "18_to_19",
                    "20",
                    "21",
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

    @classmethod
    def _get_aggregation_bins(cls, prefix="total"):
        """
        Return the aggregation bins for calculating the median age
        from the distribution. 

        Parameters
        ----------
        prefix : "total", "male", "female"
            return the column names with this prefix

        Returns
        -------
        bins : list of tuples
            tuples of (start, stop, column name)
        """
        if prefix not in ["total", "male", "female"]:
            raise ValueError("allowed prefix values are 'total', 'male', and 'female'")

        bins = []
        for i, g in enumerate(cls.GROUPS[1:]):
            if i == 0:
                start, end = 0, 4
            elif g.endswith("and_over"):
                start = float(g.split("_")[0])
                end = np.inf
            else:
                fields = g.split("_to_")
                if len(fields) == 2:
                    start, end = map(float, fields)
                else:
                    start = end = float(fields[0])

            bins.append((start, end, f"{prefix}_{g}"))

        return bins
