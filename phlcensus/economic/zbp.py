from ..regions import ZIPCodes
from ..core import Dataset, data_dir
import cenpy as cen
import pandas as pd
import collections

__all__ = ["ZBP"]


class ZBP(Dataset):
    """
    Data from the ZIP Code Business Patterns (ZBP) survey.
    """

    YEARS = [2016, 2015, 2014, 2013, 2012]

    @classmethod
    def process(cls, df):
        """
        Process the raw data files, adding custom columns if desired.
        """
        groups = collections.OrderedDict(
            {
                "universe": ["All establishments"],
                "small": [
                    "Establishments with 1 to 4 employees",
                    "Establishments with 5 to 9 employees",
                    "Establishments with 10 to 19 employees",
                ],
                "medium": ["Establishments with 20 to 49 employees"],
                "large": [
                    "Establishments with 50 to 99 employees",
                    "Establishments with 100 to 249 employees",
                    "Establishments with 250 to 499 employees",
                    "Establishments with 500 to 999 employees",
                    "Establishments with 1,000 employees or more",
                ],
            }
        )

        # Create small, medium, large tags
        out = []
        for groupset, group_list in groups.items():
            valid = df["establishment_size"].isin(group_list)
            df.loc[valid, "establishment_size"] = groupset

            N = (
                df.loc[valid]
                .groupby(["naics_code", "establishment_size", "zip_code"])[
                    "num_establishments"
                ]
                .sum()
                .reset_index()
            )
            out.append(N)
        out = pd.concat(out)

        # Merge in original data
        out = pd.merge(
            df.drop(labels=["num_establishments", "establishment_size"], axis=1),
            out,
            on=["naics_code", "zip_code"],
        ).drop_duplicates(subset=["zip_code", "naics_code", "establishment_size"])

        return out.reset_index(drop=True)

    @classmethod
    def get_path(cls, year=2016):
        return data_dir / cls.__name__ / str(year)

    @classmethod
    def download(cls, year=2016):
        return cls._query_census_api(year=year)

    @classmethod
    def get(cls, fresh=False, year=2016):
        """
        Load the dataset, optionally downloading a fresh copy.

        Parameters
        ---------
        fresh : bool, optional
            a boolean keyword that specifies whether a fresh copy of the 
            dataset should be downloaded
        year : int, optional
            the data year to download

        Returns
        -------
        data : DataFrame/GeoDataFrame
            the dataset as a pandas/geopandas object
        """
        # Verify input year
        if year not in cls.YEARS:
            raise ValueError(f"Allowed values for 'year' are: {cls.YEARS}")

        # return
        return super().get(fresh=fresh, year=year)

    @classmethod
    def _query_census_api(cls, year):
        """
        Download the requested fields from the American Community Survey
        using the Census API.

        Parameters
        ----------
        year : int
            the year of data to download
        """
        # Get the list of zip codes in philadelphia
        zipcodes = ZIPCodes.get().assign(zip_code=lambda df: df.zip_code.astype(str))
        zips = ",".join(zipcodes["zip_code"].unique())

        # initialize the API
        api = cen.remote.APIConnection(f"ZBP{year}")

        # the codes to download
        naics_codes = [
            "00",
            "23",
            "31-33",
            "42",
            "44-45",
            "48-49",
            "51",
            "52",
            "53",
            "54",
            "55",
            "56",
            "61",
            "71",
            "72",
        ]

        df = (
            pd.concat(
                [
                    api.query(
                        ["NAICS2012_TTL", "ESTAB", "EMPSZES", "EMPSZES_TTL"],
                        geo_unit=f"zipcode:{zips}",
                        **{"NAICS2012": code},
                    )
                    for code in naics_codes
                ]
            )
            .rename(
                columns={
                    "NAICS2012": "naics_code",
                    "NAICS2012_TTL": "naics_code_description",
                    "ESTAB": "num_establishments",
                    "EMPSZES_TTL": "establishment_size",
                    "zipcode": "zip_code",
                }
            )
            .drop(labels=["EMPSZES"], axis=1)
            .assign(num_establishments=lambda df: df.num_establishments.astype(int))
        )

        # merge zip codes
        df = zipcodes.merge(df, on="zip_code")

        # Return the processed data
        return cls.process(df)
