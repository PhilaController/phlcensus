from ..regions import PUMAs, CensusTracts
from ..core import Dataset, data_dir
from ..aggregate import aggregate_tracts
from .. import DEFAULT_YEAR
import census_data_aggregator as cda
import cenpy as cen
import pandas as pd
import numpy as np

DATASETS = {}


class ACSDataset(Dataset):
    """
    A base class to represent a dataset downloaded from the Census API.

    Notes
    -----
    Valid for years from 2012 to 2017.
    """

    AGGREGATION = None
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
    def get_path(cls, level="puma", year=DEFAULT_YEAR, N=5):
        return data_dir / cls.__name__ / level / f"{year}{N}YR"

    @classmethod
    def download(cls, level="puma", year=DEFAULT_YEAR, N=5):
        return cls.process(cls._query_census_api(level=level, year=year, N=N))

    @classmethod
    def get(cls, fresh=False, level="puma", year=DEFAULT_YEAR, N=5):
        """
        Load the dataset, optionally downloading a fresh copy.

        Parameters
        ---------
        fresh : bool, optional
            a boolean keyword that specifies whether a fresh copy of the 
            dataset should be downloaded
        level : str, optional
            the geographic level to return; one of "puma", "tract", "nta", or "city"
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
        allowed_levels = ["puma", "nta", "tract", "city"]
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
            the geographic level to return; one of "puma", "tract", "nta", or "city"
        year : int
            the year of data to download
        N : int, {1,5}
            pull data from the 1-year or 5-year ACS
        """
        if level == "nta":

            if cls.AGGREGATION is None:
                raise ValueError("The input data cannot be aggregated by NTA")

            # Get the data by tract
            data = cls._query_census_api(level="tract", year=year, N=N)

            # Aggregate the count data
            if cls.AGGREGATION == "count":
                return aggregate_tracts(data, "nta", "count")
            else:  # median data

                assert hasattr(cls, "median_aggregator")

                # Store the result
                result = None

                # Loop over the distribution data
                # for each specific median column we calculate
                for (dist_df, bins, output_col) in cls._median_aggregator(
                    year=year, N=N
                ):

                    # Aggregate the distribution counts to a median
                    merged = aggregate_tracts(dist_df, "nta", "median", bins=bins)

                    # Rename the median column to the right name
                    merged = merged.rename(
                        columns={
                            "median": output_col,
                            "median_moe": f"{output_col}_moe",
                        }
                    )

                    # Save the result, merging with previous results
                    if result is None:
                        result = merged
                    else:
                        cols = merged.columns.difference(result.columns)
                        result = pd.merge(
                            result,
                            merged[cols],
                            left_index=True,
                            right_index=True,
                            how="outer",
                        )

                return result.loc[:, data.columns]

        else:

            # Initialize the API
            api = cen.remote.APIConnection(f"ACSDT{N}Y{year}")

            # Format the variable names properly
            variables = {}
            for field, renamed in cls.RAW_FIELDS.items():
                old_name = f"{cls.TABLE_NAME}_{field}"
                variables[f"{old_name}E"] = renamed  # estimate
                variables[f"{old_name}M"] = f"{renamed}_moe"  # margin of error

            # Query the API
            if level == "tract":
                df = cls._query_census_by_tract(api, variables, year)
            elif level == "puma":
                df = cls._query_census_by_puma(api, variables, year)
            elif level == "city":
                df = cls._query_census_by_city(api, variables)

            # Convert columns from strings to numbers
            for col in variables:
                df[col] = pd.to_numeric(df[col])

            # Set geo_id as an integer
            if "geo_id" in df.columns:
                df["geo_id"] = df["geo_id"].astype(str)

            # Return the processed data
            return df.rename(columns=variables)

    @classmethod
    def _query_census_by_puma(cls, api, variables, year):
        """
        Query the Census by PUMA.

        Parameters
        ----------
        api : 
            the cenpy remote API connection to the census
        variables : dict
            a dict mapping Census column names to human-readable column names
        year : int
            the year of the data to query
        """
        # The PUMA boundaries
        pumas = PUMAs.get(year=year)

        # Query the census API to get the raw data
        df = (
            api.query(
                cols=list(variables),
                geo_unit="public use microdata area:*",
                geo_filter={"state": "42"},
            )
            .assign(
                geo_id=lambda df: df.apply(
                    lambda r: f"{r['state']}{r['public use microdata area']}", axis=1
                )
            )
            .drop(labels=["state", "public use microdata area"], axis=1)
        )

        # Merge with the PUMAs
        pumas["geo_id"] = pumas["geo_id"].astype(str)
        return pumas.merge(df, on="geo_id")

    @classmethod
    def _query_census_by_tract(cls, api, variables, year):
        """
        Query the Census by Census tract.

        Parameters
        ----------
        api : 
            the cenpy remote API connection to the census
        variables : dict
            a dict mapping Census column names to human-readable column names
        year : int
            the year of the data to query
        """
        # The tract boundaries
        tracts = CensusTracts.get(year=year)

        # Query the census API to get the raw data
        df = (
            api.query(
                cols=list(variables), geo_unit="tract:*", geo_filter={"state": "42"}
            )
            .assign(
                geo_id=lambda df: df.apply(
                    lambda r: f"{r['state']}{r['county']}{r['tract']}", axis=1
                )
            )
            .drop(labels=["state", "county", "tract"], axis=1)
        )

        # Merge with the tracts
        tracts["geo_id"] = tracts["geo_id"].astype(str)
        return tracts.merge(df, on="geo_id")

    @classmethod
    def _query_census_by_city(cls, api, variables):
        """
        Query the Census for Philadelphia city.

        Parameters
        ----------
        api : 
            the cenpy remote API connection to the census
        variables : dict
            a dict mapping Census column names to human-readable column names
        """
        return api.query(
            cols=list(variables), geo_unit="county:101", geo_filter={"state": "42"}
        ).drop(labels=["state", "county"], axis=1)


def approximate_sum(row, cols):
    """
    Apply this function to a DataFrame, summing over the columns specified.

    Note
    ----
    This assumes a margin of error column exists
    """
    args = [(row[col], row[f"{col}_moe"]) for col in cols]
    return pd.Series(cda.approximate_sum(*args))


def approximate_ratio(row, cols):
    """
    Apply this function to a DataFrame, taking the ratio of the columns 
    specified.

    Note
    ----
    This assumes a margin of error column exists
    """
    args = [(row[col], row[f"{col}_moe"]) for col in cols]

    if args[-1][0] == 0:
        return pd.Series([np.nan, np.nan])
    else:
        return pd.Series(cda.approximate_ratio(*args))


def calculate_cv(df):
    """
    Calculate the coefficient of variation for the input 
    ACS dataset

    Parameters
    ----------
    df : DataFrame
        the ACS data 
    """
    # Get the non-geographic cols
    cols = df.filter(regex="^(?!geo)\w*$(?<!moe)", axis=1).columns

    # Only use cols with a _moe pair
    cols = [col for col in cols if f"{col}_moe" in df.columns]

    # standard error
    SEs = df[[f"{col}_moe" for col in cols]].values / 1.645

    # values
    values = df[cols].values
    values[values == 0.0] = np.nan  # handle zeros separately

    # coefficient of variation
    CVs = SEs / values

    # output
    out = df.copy()
    out = out[[col for col in out.columns if col.startswith("geo") or col in cols]]

    # copy over CVs
    out[cols] = CVs
    return out

