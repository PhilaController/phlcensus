__version__ = "0.0.6"

from pathlib import Path

data_dir = Path(__file__).parent / "data"

EPSG = 2272

from . import acs, economic, external
from .core import DATASETS
from .regions import *

__all__ = sorted(DATASETS)


def available_datasets():
    """
    Return a list of the names of the available datasets.
    """
    return [DATASETS[cls] for cls in sorted(DATASETS)]
