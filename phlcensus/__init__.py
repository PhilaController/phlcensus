__version__ = "0.0.1"

from pathlib import Path

data_dir = Path(__file__).parent / "data"

EPSG = 2272

from .core import DATASETS
from . import acs
from . import econ
from . import external
from .regions import *
from .analysis import *

__all__ = sorted(DATASETS)


def available_datasets():
    """
    Return a list of the names of the available datasets.
    """
    return [DATASETS[cls] for cls in sorted(DATASETS)]

