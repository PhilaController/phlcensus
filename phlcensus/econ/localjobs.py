from ..core import Dataset, EPSG, data_dir
from ..regions import CensusTracts
import pandas as pd

__all__ = ["JobsByWorkTract", "JobsByHomeTract"]


class JobsDatasetBase(Dataset):
    """
    Base class for loading data from the 
    Longitudinal Employer-Household Dynamics (LEHD).

    Source
    ------
    https://lehd.ces.census.gov/
    """

    YEARS = list(range(2002, 2018))
    URL = "https://lehd.ces.census.gov/data/lodes/LODES7/pa"

    @classmethod
    def get_path(cls, year=2017):
        return data_dir / cls.__name__ / str(year)

    @classmethod
    def download(cls, key, year=2017):

        # Get the year
        if year not in cls.YEARS:
            raise ValueError(f"Valid years are: {cls.YEARS}")

        if key not in ["w_geocode", "h_geocode"]:
            raise ValueError("valid column names are 'w_geocode' or 'h_geocode'")

        # Load the cross-walk
        xwalk = pd.read_csv(f"{cls.URL}/pa_xwalk.csv.gz").assign(
            tabblk2010=lambda df: df.tabblk2010.astype(str)
        )

        # Load the Origin-Destination data
        # JT00 --> all jobs
        data = [pd.read_csv(f"{cls.URL}/od/pa_od_main_JT00_{year}.csv.gz")]
        if key == "w_geocode":
            data.append(pd.read_csv(f"{cls.URL}/od/pa_od_aux_JT00_{year}.csv.gz"))
        data = pd.concat(data).assign(
            h_geocode=lambda df: df.h_geocode.astype(str),
            w_geocode=lambda df: df.w_geocode.astype(str),
        )
        data["is_resident"] = False

        # load the tracts
        tracts = CensusTracts.get().assign(geo_id=lambda df: df.geo_id.astype(str))

        # determine residents
        residents = data["h_geocode"].str.slice(0, 11).isin(tracts["geo_id"])
        data.loc[residents, "is_resident"] = True

        # sum by block group
        cols = [col for col in data.columns if col.startswith("S")]
        groupby = [key, "is_resident"]
        N = data[groupby + cols].groupby(groupby).sum().reset_index()

        # merge with crosswalk
        N = N.merge(xwalk, left_on=key, right_on="tabblk2010", how="left").assign(
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
        if key == "w_geocode":
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

        if key == "w_geocode":
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


class JobsByWorkTract(JobsDatasetBase):
    """
    Local jobs by the census tract where the employee
    works. 

    Source
    ------
    Longitudinal Employer-Household Dynamics (LEHD) 
    https://lehd.ces.census.gov/
    """

    @classmethod
    def download(cls, **kwargs):
        return super().download("w_geocode", **kwargs)


class JobsByHomeTract(JobsDatasetBase):
    """
    Local jobs by the census tract where the employee
    lives. 

    Source
    ------
    Longitudinal Employer-Household Dynamics (LEHD) 
    https://lehd.ces.census.gov/
    """

    @classmethod
    def download(cls, **kwargs):
        return super().download("h_geocode", **kwargs)
