from ...core import Dataset, EPSG, data_dir
from ...regions import CensusTracts, NTAs, PUMAs
from ...aggregate import aggregate_tracts
from ... import crosswalk
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
    def download(cls, year=DEFAULT_YEAR, kind="w_geocode", job_type="JT00"):
        """
        Download the raw LODES data files for the Origin-Destination subset.

        Parameters
        ----------
        kind : "work" or "home"
            load data according to where the job is located ("work") or 
            where the employee lives ("home")
        year : int
            the dataset's year; available dating back to 2002
        job_type : str, optional
            the types of jobs: one of "all", "primary", "private", "private_primary"
        """
        # The main file (people who live and work in PA)
        data = [pd.read_csv(f"{cls.URL}/od/pa_od_main_{job_type}_{year}.csv.gz")]

        # Add aux file: people who work in PA but live out of state
        if kind == "w_geocode":
            data.append(pd.read_csv(f"{cls.URL}/od/pa_od_aux_{job_type}_{year}.csv.gz"))

        # Combine the data files
        data = pd.concat(data).assign(
            h_geocode=lambda df: df.h_geocode.astype(str),
            w_geocode=lambda df: df.w_geocode.astype(str),
        )

        return data

    @classmethod
    def process(cls, data, year=DEFAULT_YEAR, kind="w_geocode", level="tract"):
        """
        Process the census tract level data, optionally aggregating to a higher 
        level, e.g., neighborhood, PUMA, etc.
        """

        # Convert from geo_id from blocks to tracts
        for col in ["h_geocode", "w_geocode"]:
            data[col] = data[col].astype(str).str.slice(0, 11)

        # Value-added columns to add
        # Default is False
        value_added = ["is_resident", "work_at_home"]
        if kind == "h_geocode":
            value_added.append("work_in_center_city")
        for col in value_added:
            data[col] = False

        # load the tracts
        tracts = CensusTracts.get(year=year).assign(
            geo_id=lambda df: df.geo_id.astype(str)
        )

        # Determine city residents
        data.loc[data["h_geocode"].isin(tracts["geo_id"]), "is_resident"] = True

        # cross walk to find work in home area
        if level != "tract":

            # Get the cross walk from tracts to the desired level
            if level == "nta":
                xwalk = crosswalk.tracts_to_ntas()
            elif level == "puma":
                xwalk = crosswalk.tracts_to_pumas()
            xwalk["geo_id_tract"] = xwalk["geo_id_tract"].astype(str)

            # Merge in crosswalked areas
            for col in ["h_geocode", "w_geocode"]:
                data = data.merge(
                    xwalk[["geo_id_tract", f"geo_id_{level}"]],
                    left_on=col,
                    right_on="geo_id_tract",
                    how="left",
                )

            # Determine work in same area
            data.loc[
                data[f"geo_id_{level}_x"] == data[f"geo_id_{level}_y"], "work_at_home"
            ] = True

            # Remove extra columns
            data = data.drop(
                labels=[
                    f"geo_id_{area}_{tag}"
                    for area in ["tract", level]
                    for tag in ["x", "y"]
                ],
                axis=1,
            )

        else:
            # Determine work in home tract
            data.loc[data["h_geocode"] == data["w_geocode"], "work_at_home"] = True

        # Work in Center City?
        if kind == "h_geocode":

            # PUMA cross walk
            xwalk_pumas = crosswalk.tracts_to_pumas()

            # Get center city tracts
            in_center_city = xwalk_pumas["geo_name_puma"].str.contains("Center City")
            CC_tracts = (
                xwalk_pumas.loc[in_center_city, "geo_id_tract"].astype(str).tolist()
            )

            # Set to True
            data.loc[data["w_geocode"].isin(CC_tracts), "work_in_center_city"] = True

        # Sum by census tract
        cols = [col for col in data.columns if col.startswith("S")]
        groupby = [kind] + value_added
        N = data[groupby + cols].groupby(groupby).sum().reset_index()

        # Inner merge to trim to only Philly census tracts
        data = tracts.merge(N, left_on="geo_id", right_on=kind, how="inner")

        # Are we aggregating to a higher level??
        if level != "tract":

            # Merge
            data = (
                xwalk.merge(
                    data.drop(labels=["geometry"], axis=1),
                    left_on="geo_id_tract",
                    right_on=kind,
                    how="left",
                )
                .drop(
                    labels=["geo_id", "geo_name", "geo_name_tract", "geo_id_tract"],
                    axis=1,
                )
                .rename(
                    columns={
                        f"geo_id_{level}": "geo_id",
                        f"geo_name_{level}": "geo_name",
                    }
                )
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

            # Get the subset and sum over census tract
            sub_df = data.query(query).groupby("geo_id")[cols].sum().reset_index()

            # Merge
            out = out.merge(
                sub_df.rename(
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
                how="left",
            )

        # Add work in home tract
        out = out.merge(
            data.query("work_at_home == True")
            .groupby("geo_id")["S000"]
            .sum()
            .reset_index()
            .rename(columns={"S000": f"total_work_at_home"}),
            left_on="geo_id",
            right_on="geo_id",
            how="left",
        )

        # Add work in Center City
        if kind == "h_geocode":
            out = out.merge(
                data.query("work_in_center_city == True")
                .groupby("geo_id")["S000"]
                .sum()
                .reset_index()
                .rename(columns={"S000": f"total_work_in_center_city"}),
                left_on="geo_id",
                right_on="geo_id",
                how="left",
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

        # Sort by geo id and reset
        out = out.sort_values("geo_id").reset_index(drop=True)

        # if we aggregated, add the right geometries
        if level != "tract":
            if level == "nta":
                regions = NTAs.get()
            elif level == "puma":
                regions = PUMAs.get()

            out = regions[["geometry", "geo_id"]].merge(
                out.drop(labels=["geometry"], axis=1), on="geo_id", how="left"
            )

        # fill missing
        cols = [col for col in out.columns if not col.startswith("geo")]
        out[cols] = out[cols].fillna(0)

        return out.drop(labels=[kind], axis=1)

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

        # Get the raw census tract level data
        data = super().get(fresh=fresh, kind=kind, year=year, job_type=job_type)

        # Return processed data
        return cls.process(data, year=year, kind=kind, level=level)
