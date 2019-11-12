from ..regions import PUMAs, CensusTracts
from ..core import Dataset, data_dir
import cenpy as cen
import pandas as pd

DATASETS = {}


class ACSDataset(Dataset):
    """
    A base class to represent a dataset downloaded from the Census API.
    """

    YEARS = [2017, 2016, 2015, 2014, 2013, 2012]

    def __init_subclass__(cls, **kwargs):
        """
        Register subclasses of this class
        """
        if cls not in DATASETS:
            DATASETS[cls.__name__] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def process(cls, df):
        """
        Process the raw data files, adding custom columns if desired.
        """
        return df

    @classmethod
    def get_path(cls, level="puma", year=2017, N=5):
        return data_dir / cls.__name__ / level / f"{year}{N}YR"

    @classmethod
    def download(cls, level="puma", year=2017, N=5):
        return cls._query_census_api(level=level, year=year, N=N)

    @classmethod
    def get(cls, fresh=False, level="puma", year=2017, N=5):
        """
        Load the dataset, optionally downloading a fresh copy.

        Parameters
        ---------
        fresh : bool, optional
            a boolean keyword that specifies whether a fresh copy of the 
            dataset should be downloaded
        level : str, optional
            the geographic level to return; one of "puma", "tract", or "city"
        year : int, optional
            the data year to download
        N : int, {1,5}
            pull data from the 1-year or 5-year ACS

        Returns
        -------
        data : DataFrame/GeoDataFrame
            the dataset as a pandas/geopandas object
        """
        # Verify input level
        allowed_levels = ["puma", "tract", "city"]
        level = level.lower()
        if level not in allowed_levels:
            raise ValueError(f"Allowed values for 'level' are: {allowed_levels}")

        # Verify input year
        if year not in cls.YEARS:
            raise ValueError(f"Allowed values for 'year' are: {cls.YEARS}")

        # Verify n years
        if N not in [1, 5]:
            raise ValueError(f"Allowed values for 'N' are: [1, 5]")

        # return
        return super().get(fresh=fresh, level=level, year=year, N=N)

    @classmethod
    def _query_census_api(cls, level, year, N):
        """
        Download the requested fields from the American Community Survey
        using the Census API.

        Parameters
        ----------
        level : str
            the geographic level to return; one of "puma", "tract", or "city"
        year : int
            the year of data to download
        N : int, {1,5}
            pull data from the 1-year or 5-year ACS
        """
        # Return data by PUMA
        if level == "puma":
            boundaries = PUMAs.get(year=year)
            geo_unit = "public use microdata area"
        # Return data by tract
        elif level == "tract":
            boundaries = CensusTracts.get(year=year)
            geo_unit = "tract"
        # Return data citywide
        else:
            geo_unit = "place"
            boundaries = None

        # initialize the API
        api = cen.remote.APIConnection(f"ACSDT{N}Y{year}")

        # Format the variable names properly
        variables = {}
        for field, renamed in cls.RAW_FIELDS.items():
            old_name = f"{cls.TABLE_NAME}_{field}"
            variables[f"{old_name}E"] = renamed  # estimate
            variables[f"{old_name}M"] = f"{renamed}_moe"  # margin of error

        # Query the census API to get the raw data
        if level == "city":
            # Get data for Philadelphia county
            df = api.query(
                cols=list(variables), geo_unit="county:101", geo_filter={"state": "42"}
            )
        else:
            df = api.query(
                cols=list(variables),
                geo_unit=f"{geo_unit}:*",
                geo_filter={"state": "42"},
            )

        # Create the ID column and remove unnecessary columns
        if level == "puma":
            df["geo_id"] = df.apply(
                lambda r: f"{r['state']}{r['public use microdata area']}", axis=1
            )
            df = df.drop(labels=["state", "public use microdata area"], axis=1)
        elif level == "tract":
            df["geo_id"] = df.apply(
                lambda r: f"{r['state']}{r['county']}{r['tract']}", axis=1
            )
            df = df.drop(labels=["state", "county", "tract"], axis=1)

        if level != "city":
            # set the boundary ID as a string
            boundaries["geo_id"] = boundaries["geo_id"].astype(str)

            # Merge the data with the boundaries
            df = boundaries.merge(df, on="geo_id")
        else:
            df = df.drop(labels=["state", "county"], axis=1)

        # Convert columns from strings to numbers
        for col in variables:
            df[col] = pd.to_numeric(df[col])

        # Return the processed data
        return cls.process(df.rename(columns=variables))
