from . import data_dir, EPSG
from abc import ABC, abstractclassmethod
import geopandas as gpd
import pandas as pd
import json

DATASETS = {}


class Dataset(ABC):
    """
    An abstract base class to represent a dataset.
    """

    date_columns = []

    def __init_subclass__(cls, **kwargs):
        """
        Register subclasses of this class
        """
        if cls not in DATASETS:
            DATASETS[cls.__name__] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def _format_data(cls, data):
        """
        An internal method to format the input dataframe. The performs 
        the following tasks:

        1. Convert from a pandas DataFrame to a geopandas GeoDataFrame, if possible
        2. Convert date columns to pandas Datetime objects

        It returns the formatted DataFrame/GeoDataFrame.
        """
        # convert to GeoDataFrame
        if "geometry" in data.columns:
            from shapely import wkt

            data.geometry = data.geometry.apply(wkt.loads)
            data = gpd.GeoDataFrame(
                data, geometry="geometry", crs={"init": f"epsg:{EPSG}"}
            )

        # convert date columns
        for col in cls.date_columns:
            data[col] = pd.to_datetime(data[col])

        return data

    @classmethod
    def meta(cls):
        """
        Dictionary of meta-data related to the dataset.
        """
        path = data_dir / cls.__name__ / "meta.json"
        if path.exists():
            return json.load(path.open(mode="r"))
        else:
            return {}

    @classmethod
    def now(cls):
        """
        Return the current datetime as a pandas Datetime object.
        """
        return str(pd.datetime.now())

    @classmethod
    def get_path(cls, **kwargs):
        """
        Return the directory path holding the data.
        """
        return data_dir / cls.__name__

    @classmethod
    def get(cls, fresh=False, **kwargs):
        """
        Load the dataset, optionally downloading a fresh copy.

        Parameters
        ---------
        fresh : bool, optional
            a boolean keyword that specifies whether a fresh copy of the 
            dataset should be downloaded
        **kwargs : 
            Additional keywords are passed to the `get_path()` function and 
            the `download()` function

        Returns
        -------
        data : DataFrame/GeoDataFrame
            the dataset as a pandas/geopandas object
        """
        # Get the folder path
        dirname = cls.get_path(**kwargs)
        if not dirname.exists():
            dirname.mkdir(parents=True)
            fresh = True

        # Download and save a fresh copy
        data_path = dirname / "data.csv"
        meta_path = dirname / "meta.json"
        if not data_path.exists() or fresh:

            # download and save a fresh copy
            data = cls.download(**kwargs)
            data.to_csv(data_path, index=False)

            # save the download time
            meta = {"download_time": cls.now()}
            json.dump(meta, meta_path.open(mode="w"))

        # Load and return the formatted data
        return cls._format_data(pd.read_csv(data_path, low_memory=False))

    @abstractclassmethod
    def download(cls, **kwargs):
        """
        Download and return the dataset.

        This must be defined for subclasses of the `Dataset` class.

        Returns
        -------
        data : DataFrame/GeoDataFrame
            the dataset as a data frame object
        """
        raise NotImplementedError

