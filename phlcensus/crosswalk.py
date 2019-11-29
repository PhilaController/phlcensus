from . import regions
import geopandas as gpd
from functools import lru_cache

__all__ = ["tracts_to_pumas", "tracts_to_ntas", "ntas_to_pumas"]


def _get_crosswalk(left, right, lsuffix, rsuffix):
    """
    Internal function to generate a cross walk data frame
    between two sets of geometries.

    Notes
    -----
    The ``left`` data frame should hold the more granular geometry, 
    e.g., ``left`` is within ``right``
    """
    # get the intersection
    # NNif the smaller geometry intersects multiple larger geometries,
    # only keep the largest intersection
    i = (
        gpd.overlay(left, right)
        .assign(area=lambda df: df.geometry.area)
        .sort_values(["geo_id_1", "area"], ascending=[True, False])
        .drop_duplicates("geo_name_1", keep="first")
        .rename(
            columns={
                "geo_id_1": "geo_id",
                "geo_id_2": f"geo_id_{rsuffix}",
                "geo_name_2": f"geo_name_{rsuffix}",
            }
        )
    )

    # do the cross walk
    xwalk = left.merge(
        i[["geo_id", f"geo_id_{rsuffix}", f"geo_name_{rsuffix}"]], on="geo_id"
    )

    # rename and return
    return xwalk.rename(
        columns={"geo_id": f"geo_id_{lsuffix}", "geo_name": f"geo_name_{lsuffix}"}
    )


@lru_cache(maxsize=128)
def tracts_to_pumas():
    """
    Return the crosswalk between census tracts and PUMAS.
    """

    return _get_crosswalk(
        regions.CensusTracts.get(), regions.PUMAs.get(), "tract", "puma"
    )


@lru_cache(maxsize=128)
def ntas_to_pumas():
    """
    Return the crosswalk between Neighborhood Tabulation Areas (NTAs) and PUMAS.
    """
    return _get_crosswalk(regions.NTAs.get(), regions.PUMAs.get(), "nta", "puma")


@lru_cache(maxsize=128)
def tracts_to_ntas():
    """
    Return the crosswalk between Neighborhood Tabulation Areas (NTAs) and PUMAS.
    """
    return _get_crosswalk(
        regions.CensusTracts.get(), regions.NTAs.get(), "tract", "nta"
    )

