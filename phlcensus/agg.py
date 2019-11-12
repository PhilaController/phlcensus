import census_data_aggregator as cda
import pandas as pd
import numpy as np


def approximate_sum(row, cols):
    args = [(row[col], row[f"{col}_moe"]) for col in cols]
    return pd.Series(cda.approximate_sum(*args))


def approximate_ratio(row, cols):
    args = [(row[col], row[f"{col}_moe"]) for col in cols]

    if args[-1][0] == 0:
        return pd.Series([np.nan, np.nan])
    else:
        return pd.Series(cda.approximate_ratio(*args))


def approximate_proportion(row, cols):
    args = [(row[col], row[f"{col}_moe"]) for col in cols]

    if args[-1][0] == 0:
        return pd.Series([np.nan, np.nan])
    else:
        return pd.Series(cda.approximate_proportion(*args))
