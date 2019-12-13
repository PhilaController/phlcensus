import census_data_aggregator as cda
import pandas as pd
import geopandas as gpd
import numpy as np
from . import crosswalk


def aggregate_tracts(data, level, kind, bins=None):
    """
    Aggregate the input data, defined at the tract level, to 
    either the PUMA or NTA level.

    Parameters
    ----------
    data : DataFrame
        the data defined at the census tract level that we want to aggregate
    level : 'puma' or 'nta'
        the level we are aggregating to
    kind : 'count' or 'median'
        the kind of data we aggregating

    """
    assert level in ["nta", "puma"]
    assert kind in ["count", "median"]

    # Get the cross walk from tracts to the desired level
    if level == "nta":
        xwalk = crosswalk.tracts_to_ntas()
    elif level == "puma":
        xwalk = crosswalk.tracts_to_pumas()

    # Aggregate if we need to
    if level != "tract":

        # convert to strings
        xwalk["geo_id_tract"] = xwalk["geo_id_tract"].astype(str)
        data["geo_id"] = data["geo_id"].astype(str)

        # Merge with crosswalk
        merged = xwalk.merge(
            data.drop(labels=["geometry"], axis=1),
            left_on="geo_id_tract",
            right_on="geo_id",
            how="left",
        )

        # Aggregate count data
        if kind == "count":
            data = aggregate_count_data(
                merged, f"geo_id_{level}", id_vars=[f"geo_name_{level}"]
            )
        else:
            data = aggregate_median_data(
                merged, bins, f"geo_id_{level}", id_vars=[f"geo_name_{level}"]
            )

        # Rename the columns
        data = data.rename(
            columns={f"geo_id_{level}": "geo_id", f"geo_name_{level}": "geo_name"}
        )

        return data


def aggregate_median_data(df, bins, groupby, id_vars=[]):
    """
    Aggregate all columns in the input data frame, assuming
    the data is "median" data.

    Note
    ----
    The geometry of the returned object is aggregated geometry
    of all input geometries (the unary union).

    Parameters
    ----------
    df : GeoDataFrame
        the input data to aggregate
    by : str
        the name of the column that specifies the aggregation groups

    Examples
    --------
    >>> bins = cp_data.HouseholdIncome.get_aggregation_bins()
    >>> cp_data.census.aggregate_median_data(df, bins, "cluster_label", "median_income")
    
    Returns
    -------
    out : GeoDataFrame
        the output data with aggregated data and margin of error columns, 
        and the aggregated geometry polygon 
    """
    # Make sure we have the column we are grouping by
    if groupby not in df.columns:
        raise ValueError(
            f"the specified column to group by '{groupby}' is not in the input data"
        )

    # these are the column names for each bin
    # FORMAT of bins is (min, max, column_name)
    columns = [b[-1] for b in bins]

    # Make sure all of the specified columns are present
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"the specified column '{col}' is not in the input data")
        if f"{col}_moe" not in df.columns:
            raise ValueError(
                f"the specified column '{col}_moe' is not in the input data"
            )

    def _aggregate(group_df, sampling_percentage=5 * 2.5):
        """
        The function that aggregates each group
        """
        out = {}
        dist = []
        total_count = 0
        for i, col in enumerate(columns):

            n = group_df[col].sum()
            total_count += n
            dist.append(dict(min=bins[i][0], max=bins[i][1], n=n))

        # only aggregate if we have data!
        if total_count:
            aggval, moe = cda.approximate_median(
                dist, sampling_percentage=sampling_percentage
            )
        else:
            aggval = np.nan
            moe = np.nan

        result = {}
        result["median"] = aggval
        result["median_moe"] = moe
        result["geometry"] = group_df.geometry.unary_union

        return pd.Series(result)

    # this is the aggregated data, with index of "by", e.g., group label
    agg_df = df.groupby(groupby).apply(_aggregate)

    # Return a GeoDataFrame
    out = gpd.GeoDataFrame(agg_df, geometry="geometry", crs=df.crs).reset_index()

    # Add in any id variables from
    if len(id_vars):
        if groupby not in id_vars:
            id_vars.append(groupby)
        out = out.merge(df[id_vars], on=groupby).drop_duplicates(subset=[groupby])

    return out


def aggregate_count_data(df, groupby, id_vars=[]):
    """
    Aggregate all columns in the input data frame, assuming
    the data is "count" data that can be summed.

    Note
    ----
    The geometry of the returned object is aggregated geometry
    of all input geometries (the unary union).

    Parameters
    ----------
    df : GeoDataFrame
        the input data to aggregate
    groupby : str
        the name of the column that specifies the aggregation groups
    
    Returns
    -------
    out : GeoDataFrame
        the output data with aggregated data and margin of error columns, 
        and the aggregated geometry polygon 
    """
    # Make sure we have the column we are grouping by
    if groupby not in df.columns:
        raise ValueError(
            f"the specified column to group by '{by}' is not in the input data"
        )

    # data columns
    data_columns = [col for col in df.columns if not col.startswith("geo")]

    # fill na with 0
    df = df.copy()
    df[data_columns] = df[data_columns].fillna(0)

    def _aggregate(group_df):
        """
        The function that aggregates each group
        """
        out = {}
        for col in data_columns:

            # The name of the error column (if it exists)
            error_col = f"{col}_moe"

            # column values, margin of error (if it exists)
            args = np.column_stack(
                [group_df[col], group_df.get(error_col, np.zeros(len(group_df)))]
            )

            # do the aggregation
            aggval, moe = cda.approximate_sum(*args)

            # store
            out[col] = aggval
            if error_col in group_df.columns:
                out[f"{col}_moe"] = moe

        out["geometry"] = group_df.geometry.unary_union
        return pd.Series(out)

    # this is the aggregated data, with index of "by", e.g., group label
    agg_df = df.groupby(groupby).apply(_aggregate)

    # Return a GeoDataFrame
    out = gpd.GeoDataFrame(agg_df, geometry="geometry", crs=df.crs).reset_index()

    # Add in any id variables from
    if len(id_vars):
        if groupby not in id_vars:
            id_vars.append(groupby)
        out = out.merge(df[id_vars], on=groupby).drop_duplicates(subset=[groupby])

    return out

