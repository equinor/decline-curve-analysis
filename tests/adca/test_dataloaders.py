"""
Tests for dataloaders.
"""

import pandas as pd
import pytest

from dca.adca.dataloaders import load_file
from dca.datasets.dummy_data import generate_dummy_data


def test_that_creating_and_loading_dummy_data_does(tmp_path):
    """Tests that generating data works, that saving it and loading with works,
    and that the result is the same after saving and loading from disk."""

    dummy_kwargs = {
        "phases": ["oil"],
        "frequency": "monthly",
        "period_range": [None, None],
        "format": None,
        "well_ids": ["well1", "well2", "well3"],
    }

    # Generate random data
    df_generated = generate_dummy_data(table=None, **dummy_kwargs)

    # Save data
    filename = tmp_path / "production.csv"
    df_generated.to_csv(filename, index=False)

    # Now load it back via the loader
    df_loaded = load_file(table=str(filename), **dummy_kwargs)

    # Check equality
    pd.testing.assert_frame_equal(df_generated, df_loaded)


if __name__ == "__main__":
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
