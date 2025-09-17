"""
This file contains functions for loading datasets.
"""

import os
from pathlib import Path

import pandas as pd

DATASET_DIRECTORY = Path(__file__).parent


def load_monthly_sodir_production():
    """Load a dataset with monthly production data.

    - https://factpages.sodir.no/en/field/TableView/Production/Saleable/Monthly
    - Norwegian Licence for Open Government Data: https://data.norge.no/nlod/en

    """
    return pd.read_csv(os.path.join(DATASET_DIRECTORY, "field_production_monthly.csv"))
