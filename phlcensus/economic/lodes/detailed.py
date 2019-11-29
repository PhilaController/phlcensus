from ...core import Dataset, data_dir
from ...regions import CensusTracts
from ...aggregate import aggregate_tracts
from ... import DEFAULT_YEAR
import pandas as pd
import collections

__all__ = ["DetailedLODES"]


class DetailedLODES(Dataset):
    """
    Class for loading data Worker/Residence Area Characteristics
    data from LODES.

    Data can be loaded based on the census tract where the 
    employee works ("work") or where the employee lives ("home").

    This data includes additional information on jobs by industry, 
    race, and firm size.

    Source
    ------
    https://lehd.ces.census.gov/
    https://lehd.ces.census.gov/data/lodes/LODES7/LODESTechDoc7.3.pdf
    """

    YEARS = list(range(2002, 2018))
    URL = "https://lehd.ces.census.gov/data/lodes/LODES7/pa"

    RAW_FIELDS = collections.OrderedDict(
        {
            "C000": "total_jobs",
            "CA01": "age_29_or_younger",
            "CA02": "age_30_to_54",
            "CA03": "age_55_or_older",
            "CE01": "wages_1250_or_less",
            "CE02": "wages_1251_to_3333",
            "CE03": "wages_3334_or_more",
            "CNS01": "agriculture_etc",
            "CNS02": "mining_etc",
            "CNS03": "utilities",
            "CNS04": "construction",
            "CNS05": "manufacturing",
            "CNS06": "wholesale_trade",
            "CNS07": "retail_trade",
            "CNS08": "transportation_warehousing",
            "CNS09": "information",
            "CNS10": "finance_insurance",
            "CNS11": "real_estate",
            "CNS12": "technical_services",
            "CNS13": "management",
            "CNS14": "waste_management",
            "CNS15": "educational_services",
            "CNS16": "healthcare_social_services",
            "CNS17": "arts_entertainment_recreation",
            "CNS18": "accommodation_food_services",
            "CNS19": "other_services",
            "CNS20": "public_administration",
            "CR01": "white_alone",
            "CR02": "black_alone",
            "CR03": "american_indian_and_alaska_native",
            "CR04": "asian_alone",
            "CR05": "native_hawaiian_and_pacific_islander",
            "CR07": "two_or_more_races",
            "CT01": "not_latino",
            "CT02": "latino_alone",
            "CD01": "less_than_high_school",
            "CD02": "high_school_graduate",
            "CD03": "some_college",
            "CD04": "bachelors_or_higher",
            "CS01": "total_jobs_male",
            "CS02": "total_jobs_female",
            "CFA01": "firm_age_0_to_1",
            "CFA02": "firm_age_2_to_3",
            "CFA03": "firm_age_4_to_5",
            "CFA04": "firm_age_6_to_10",
            "CFA05": "firm_age_11_or_more",
            "CFS01": "firm_size_0_to_19",
            "CFS02": "firm_size_20_to_49",
            "CFS03": "firm_size_50_to_249",
            "CFS04": "firm_size_250_to_499",
            "CFS05": "firm_size_500_or_more",
        }
    )

    @classmethod
    def get_path(cls, year=DEFAULT_YEAR, kind="work", job_type="all"):
        return data_dir / cls.__name__ / kind / str(year) / job_type

    @classmethod
    def download(cls, year=DEFAULT_YEAR, kind="work", job_type="all"):

        # Validate the input year
        if year not in cls.YEARS:
            raise ValueError(f"Valid years are: {cls.YEARS}")

        # Validate the job type
        allowed_job_types = {
            "all": "JT00",
            "primary": "JT01",
            "private": "JT02",
            "private_primary": "JT03",
        }
        if job_type not in allowed_job_types:
            values = list(allowed_job_types)
            raise ValueError(f"Allowed values for 'job_type': {values}")
        job_type = allowed_job_types[job_type]

        # Validate the kind
        allowed_kinds = {"work": "wac", "home": "rac"}
        if kind not in allowed_kinds:
            values = list(allowed_kinds)
            raise ValueError(f"Allowed values for 'kind': {values}")
        kind = allowed_kinds[kind]

        # Load the data
        # See: https://lehd.ces.census.gov/data/lodes/LODES7/LODESTechDoc7.3.pdf
        filename = f"{cls.URL}/{kind}/pa_{kind}_S000_{job_type}_{year}.csv.gz"
        data = (
            pd.read_csv(filename)
            .rename(columns={"h_geocode": "geo_id", "w_geocode": "geo_id"})
            .assign(geo_id=lambda df: df.geo_id.astype(str).str.slice(0, 11))
        )

        # load the tracts
        tracts = CensusTracts.get(year=year).assign(
            geo_id=lambda df: df.geo_id.astype(str)
        )

        # sum by block group
        cols = [col for col in data.columns if col in cls.RAW_FIELDS]
        N = data[["geo_id"] + cols].groupby(["geo_id"]).sum().reset_index()

        # Initialize the output array -> one row per census tract
        out = tracts.merge(N, on="geo_id").rename(columns=cls.RAW_FIELDS)

        # Remove columns that are all zeros (missing data)
        out = out[out.columns[~(out == 0).all()]]

        return out.sort_values("geo_id").reset_index(drop=True)

    @classmethod
    def get(
        cls, fresh=False, kind="work", year=DEFAULT_YEAR, level="tract", job_type="all"
    ):
        """
        Load the dataset, optionally downloading a fresh copy.

        Parameters
        ---------
        fresh : bool, optional
            a boolean keyword that specifies whether a fresh copy of the 
            dataset should be downloaded
        kind : "work" or "home"
            load data according to where the job is located ("work") or 
            where the employee lives ("home")
        year : int
            the dataset's year; available dating back to 2002
        level : str, optional
            the geographic level, one of 'tract', 'nta', or 'puma'
        job_type : str, optional
            the types of jobs: one of "all", "primary", "private", "private_primary"
        """
        # Validate the level
        allowed = ["tract", "nta", "puma"]
        if level not in allowed:
            raise ValueError(f"Allowed values for 'level': {allowed}")

        # Get the census tract level data
        data = super().get(fresh=fresh, kind=kind, year=year, job_type=job_type)

        # Aggregate if we need to
        if level != "tract":
            data = aggregate_tracts(data, level, "count")

        # Return
        return data

