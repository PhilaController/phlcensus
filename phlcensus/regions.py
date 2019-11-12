import esri2gpd
import cenpy as cen
import geopandas as gpd
from . import EPSG, data_dir
from .core import Dataset

__all__ = ["PlanningDistricts", "CensusTracts", "PUMAs"]


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
    def get_path(cls, year=2017):
        return data_dir / cls.__name__ / str(year)

    @classmethod
    def download(cls, **kwargs):
        """
        Download the census tract boundaries
        """
        # Get the year
        YEAR = kwargs.get("year", 2017)

        # the map server layer for Census Tracts
        LAYER = 8

        # trim to PA (42) and Philadelphia County (101)
        WHERE = "STATE=42 AND COUNTY=101"

        return (
            esri2gpd.get(
                f"http://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS{YEAR}/MapServer/{LAYER}",
                where=WHERE,
                fields=["GEOID", "NAME"],
            )
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
    def get_path(cls, year=2017):
        return data_dir / cls.__name__ / str(year)

    @classmethod
    def download(cls, **kwargs):

        # Get the year
        YEAR = kwargs.get("year", 2017)

        # the map server layer for PUMAS
        LAYER = 0

        # trim to PA (42) and Philadelphia (contains 032)
        WHERE = "STATE=42 AND PUMA LIKE '%032%'"

        return (
            esri2gpd.get(
                f"http://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS{YEAR}/MapServer/{LAYER}",
                where=WHERE,
                fields=["GEOID", "NAME"],
            )
            .rename(columns={"GEOID": "geo_id", "NAME": "geo_name"})
            .sort_values("geo_id")
            .reset_index(drop=True)
            .to_crs(epsg=EPSG)
        )

