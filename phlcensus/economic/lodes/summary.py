from ...core import Dataset, EPSG, data_dir
from ...regions import CensusTracts
from ...aggregate import aggregate_tracts
from ... import DEFAULT_YEAR
import pandas as pd

__all__ = ["SummaryLODES"]


class SummaryLODES(Dataset):
    """
    Class for loading data from the Longitudinal
    Employer-Household Dynamics (LEHD) Origin-Destination
    Employment Statistics (LODES) dataset.

    This loads data from the Origin-Destination (OD) data
    files associated with LODES.

    Source
    ------
    https://lehd.ces.census.gov/
    https://lehd.ces.census.gov/data/lodes/LODES7/LODESTechDoc7.3.pdf
    """

    YEARS = list(range(2002, 2018))
    URL = "https://lehd.ces.census.gov/data/lodes/LODES7/pa"

    @classmethod
    def get_path(cls, year=DEFAULT_YEAR, kind="work", job_type="all"):
        return data_dir / cls.__name__ / kind / str(year) / job_type

    @classmethod
    def download(cls, year=DEFAULT_YEAR, kind="work", job_type="all"):

        # Get the year
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
        allowed_kinds = {"work": "w_geocode", "home": "h_geocode"}
        if kind not in allowed_kinds:
            values = list(allowed_kinds)
            raise ValueError(f"Allowed values for 'kind': {values}")
        kind = allowed_kinds[kind]

        # Load the cross-walk
        xwalk = pd.read_csv(f"{cls.URL}/pa_xwalk.csv.gz").assign(
            tabblk2010=lambda df: df.tabblk2010.astype(str)
        )

        # Load the Origin-Destination data
        # JT00 --> all jobs
        data = [pd.read_csv(f"{cls.URL}/od/pa_od_main_{job_type}_{year}.csv.gz")]
        if kind == "w_geocode":
            data.append(pd.read_csv(f"{cls.URL}/od/pa_od_aux_{job_type}_{year}.csv.gz"))
        data = pd.concat(data).assign(
            h_geocode=lambda df: df.h_geocode.astype(str),
            w_geocode=lambda df: df.w_geocode.astype(str),
        )
        data["is_resident"] = False

        # load the tracts
        tracts = CensusTracts.get(year=year).assign(
            geo_id=lambda df: df.geo_id.astype(str)
        )

        # determine residents
        residents = data["h_geocode"].str.slice(0, 11).isin(tracts["geo_id"])
        data.loc[residents, "is_resident"] = True

        # sum by block group
        cols = [col for col in data.columns if col.startswith("S")]
        groupby = [kind, "is_resident"]
        N = data[groupby + cols].groupby(groupby).sum().reset_index()

        # merge with crosswalk
        N = N.merge(xwalk, left_on=kind, right_on="tabblk2010", how="left").assign(
            trct=lambda df: df.trct.astype(str)
        )

        # Sum over tracts
        groupby = ["trct", "is_resident"]
        data = tracts.merge(
            N[groupby + cols].groupby(groupby).sum().reset_index(),
            left_on="geo_id",
            right_on="trct",
        )

        # combine resident and non-resident
        # if we are doing home tracts, everyone is a resident
        queries = ["is_resident == True"]
        tags = ["resident"]
        if kind == "w_geocode":
            queries.append("is_resident == False")
            tags.append("nonresident")

        # Initialize the output array -> one row per census tract
        out = (
            data.filter(regex="geo\w+", axis=1)
            .drop_duplicates(subset=["geo_id"])
            .reset_index(drop=True)
        )

        # add in non/resident columns
        for tag, query in zip(tags, queries):
            out = out.merge(
                data.query(query)[["geo_id"] + cols].rename(
                    columns={
                        "S000": f"{tag}_total",
                        "SA01": f"{tag}_29_or_younger",
                        "SA02": f"{tag}_30_to_54",
                        "SA03": f"{tag}_55_or_older",
                        "SE01": f"{tag}_1250_or_less",
                        "SE02": f"{tag}_1251_to_3333",
                        "SE03": f"{tag}_3334_or_more",
                        "SI01": f"{tag}_goods_producing",
                        "SI02": f"{tag}_trade_transpo_utilities",
                        "SI03": f"{tag}_all_other_industries",
                    }
                ),
                left_on="geo_id",
                right_on="geo_id",
            )

        if kind == "w_geocode":
            out["total"] = out[["resident_total", "nonresident_total"]].sum(axis=1)

            groups = [
                "29_or_younger",
                "30_to_54",
                "55_or_older",
                "1250_or_less",
                "1251_to_3333",
                "3334_or_more",
                "goods_producing",
                "trade_transpo_utilities",
                "all_other_industries",
            ]
            # Calculate totals for resident + nonresident
            for g in groups:
                cols = [f"{tag}_{g}" for tag in ["resident", "nonresident"]]
                out[f"total_{g}"] = out[cols].sum(axis=1)

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
