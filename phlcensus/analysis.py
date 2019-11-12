import numpy as np
import census_data_aggregator as cda
import geopandas as gpd
import pandas as pd

__all__ = ["calculate_cv", "aggregate_count_data", "aggregate_median_data"]


def calculate_cv(df):
    """
    Calculate the coefficient of variation for the input 
    Census data
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


def aggregate_median_data(df, bins, groupby, output_column):
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
        for i, col in enumerate(columns):
            dist.append(dict(min=bins[i][0], max=bins[i][1], n=group_df[col].sum()))

        aggval, moe = cda.approximate_median(
            dist, sampling_percentage=sampling_percentage
        )

        result = {}
        result[output_column] = aggval
        result[f"{output_column}_moe"] = moe
        result["geometry"] = group_df.geometry.unary_union

        return pd.Series(result)

    # this is the aggregated data, with index of "by", e.g., group label
    agg_df = df.groupby(groupby).apply(_aggregate)

    # Return a GeoDataFrame
    return gpd.GeoDataFrame(agg_df, geometry="geometry", crs=df.crs)


def aggregate_count_data(df, groupby):
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

    def _aggregate(group_df):
        """
        The function that aggregates each group
        """
        cols = [col for col in group_df.columns if f"{col}_moe" in group_df.columns]
        out = {}
        for col in cols:
            aggval, moe = cda.approximate_sum(*group_df[[col, f"{col}_moe"]].values)
            out[col] = aggval
            out[f"{col}_moe"] = moe

        out["geometry"] = group_df.geometry.unary_union
        return pd.Series(out)

    # this is the aggregated data, with index of "by", e.g., group label
    agg_df = df.groupby(groupby).apply(_aggregate)

    # Return a GeoDataFrame
    return gpd.GeoDataFrame(agg_df, geometry="geometry", crs=df.crs)

