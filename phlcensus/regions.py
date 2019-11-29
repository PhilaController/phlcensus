import esri2gpd
import cenpy as cen
import geopandas as gpd
import pandas as pd
from . import EPSG, data_dir, DEFAULT_YEAR
from .core import Dataset

__all__ = ["PlanningDistricts", "CensusTracts", "PUMAs", "ZIPCodes", "NTAs"]


class ZIPCodes(Dataset):
    """
    Polygons representing Philadelphia's ZIP codes.

    Notes
    -----
    These are from the 2018 Census ZIP Code Tabulation Areas (ZCTAs).

    Source
    ------
    https://phl.maps.arcgis.com/home/item.html?id=ab9d26be1df8486c8d5d706fb32b33d5
    """

    @classmethod
    def download(cls, **kwargs):

        url = "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Philadelphia_ZCTA_2018/FeatureServer/0"
        return esri2gpd.get(url, fields=["zip_code"]).to_crs(epsg=EPSG)


class PlanningDistricts(Dataset):
    """
    Planning districts in the city of Philadelphia.

    Source
    ------
    https://www.opendataphilly.org/dataset/planning-districts
    """

    @classmethod
    def download(cls, **kwargs):

        url = "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Planning_Districts/FeatureServer/0"
        return esri2gpd.get(url).to_crs(epsg=EPSG)


class CensusTracts(Dataset):
    """
    The boundary regions for census tracts in Philadelphia 
    from the 2010 Census.
    """

    @classmethod
    def get_path(cls, year=DEFAULT_YEAR):
        return data_dir / cls.__name__ / str(year)

    @classmethod
    def download(cls, **kwargs):
        """
        Download the census tract boundaries
        """
        # Get the year
        YEAR = kwargs.get("year", 2017)

        # trim to PA (42) and Philadelphia County (101)
        WHERE = "STATE=42 AND COUNTY=101"

        if YEAR >= 2010:
            URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/Census2010/tigerWMS_Census2010/MapServer/10"
        elif YEAR >= 2000:
            URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/Census2010/tigerWMS_Census2000/MapServer/6"
        else:
            raise ValueError("'year' must be greater than or equal to 2000")

        return (
            esri2gpd.get(URL, where=WHERE, fields=["GEOID", "NAME"])
            .rename(columns={"GEOID": "geo_id", "NAME": "geo_name"})
            .sort_values("geo_id")
            .reset_index(drop=True)
            .to_crs(epsg=EPSG)
        )


class PUMAs(Dataset):
    """
    The boundary regions for the Public Use Microdata Areas (PUMAs) 
    in Philadelphia from the 2010 Census.

    Source
    ------
    https://www.census.gov/programs-surveys/geography/guidance/geo-areas/pumas.html
    """

    @classmethod
    def get_path(cls, year=DEFAULT_YEAR):
        return data_dir / cls.__name__ / str(year)

    @classmethod
    def download(cls, **kwargs):

        # Get the year
        YEAR = kwargs.get("year", 2017)

        if YEAR >= 2010:
            URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/Census2010/tigerWMS_Census2010/MapServer/0"
            WHERE = "STATE=42 AND PUMA LIKE '%032%'"  # trim to philadelphia
        elif YEAR >= 2000:
            URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/Census2010/tigerWMS_Census2000/MapServer/0"
            WHERE = "STATE=42 AND PUMA LIKE '%041%'"  # trim to philadelphia
        else:
            raise ValueError("'year' must be greater than or equal to 2000")

        return (
            esri2gpd.get(URL, where=WHERE, fields=["GEOID", "NAME"])
            .rename(columns={"GEOID": "geo_id", "NAME": "geo_name"})
            .sort_values("geo_id")
            .reset_index(drop=True)
            .to_crs(epsg=EPSG)
        )


class NTAs(Dataset):
    """
    Neighborhood Tabulation Areas (NTAs), as determined from a sensible 
    aggregation of Census tracts.
    """

    @classmethod
    def get_path(cls):
        return data_dir / cls.__name__

    @classmethod
    def download(cls, **kwargs):

        YEAR = kwargs.get("year", 2017)
        if YEAR < 2010:
            raise ValueError(
                "Neighborhood Tabulation Areas are only defined for 2010 Census tracts"
            )
        return pd.read_csv(data_dir / cls.__name__ / "data_raw.csv")
