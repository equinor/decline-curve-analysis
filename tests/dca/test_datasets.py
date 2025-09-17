"""
Test dataset functions.
"""

from dca.datasets import load_monthly_sodir_production


def test_that_datasets_load():
    df = load_monthly_sodir_production()
    assert not df.empty
