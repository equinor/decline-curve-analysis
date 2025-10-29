import os
import pathlib
import string
import subprocess

import pytest

YAML_EXAMPLE = string.Template(
    """
# A group is a collection of wells run together.
# One asset can have several groups, e.g. 'VolveGas', 'VolveOilShortTerm', etc
- group:
  name: "SoDir_Oil"
  
  # Data source settings
  source:
    name: "sodir"
    phases: ["oil"]
    frequency: "monthly"
    
  # The wells included in this group. They can be split into segments.
  # These are the top fields (NOT WELLS) from sodir in terms of length
  wells:
    "DRAUGEN":
      - ["2000-01", null]
    "GYDA":
    "SYGNA,SNORRE":
      - ["2008-01", null]
    
  # how to preprocess data: calendar days vs. producing days, etc
  preprocessing: ${preprocessing}
    
  curve_fitting:
    # How to split the times series for (1) out-of-sample error reporting and
    # (2) hyperparameter tuning.
    # float -> split time series by fraction (between 0 and 1)
    # int -> split time series by time index (negative or positive)
    # date -> split time series by this date (format "2024-01-25")
    split : ${split}
  
    # What kind of decline curve to use. Options: "arps" or "exponential"
    curve_model : "arps"
  
    # in units of input resolution, e.g., months or days
    forecast_periods : ${forecast_periods}

  # Hyperparameters given as numbers are fixed, and those given as
  # a range [low, high] are tuned out-of-sample on test data using the split
  hyperparameters:
    half_life : [12, 120]
    prior_strength : [0.01, 10]
"""
)


def partial_product(*iterables):
    """Yields tuples containing one item from each iterator, with subsequent
    tuples changing a single item at a time by advancing each iterator until it
    is exhausted. This sequence guarantees every value in each iterable is
    output at least once without generating all possible combinations.

    This may be useful, for example, when testing an expensive function.

        >>> list(partial_product('AB', 'C', 'DEF'))
        [('A', 'C', 'D'), ('B', 'C', 'D'), ('B', 'C', 'E'), ('B', 'C', 'F')]
    """
    # https://github.com/more-itertools/more-itertools/blob/8756668e439f5b5e15ed457f7c292f0bd975043d/more_itertools/more.py#L4831

    iterators = list(map(iter, iterables))

    try:
        prod = [next(it) for it in iterators]
    except StopIteration:
        return
    yield tuple(prod)

    for i, it in enumerate(iterators):
        for prod[i] in it:
            yield tuple(prod)


@pytest.mark.parametrize(
    "split, preprocessing, forecast_periods",
    partial_product(
        ["0.8", "-6", "2010-01"], ["calendar_time", "producing_time"], [120, "2031-01"]
    ),
)
def test_CLI(tmp_path, split, preprocessing, forecast_periods):
    """Smoketest for CLI."""
    # Substitute the template with strings
    yaml_example = YAML_EXAMPLE.substitute(
        preprocessing=preprocessing, split=split, forecast_periods=forecast_periods
    )

    # Create a config file
    tmp_path.mkdir(exist_ok=True)
    yaml_file = pathlib.Path(tmp_path / "test_config.yaml")
    os.chdir(tmp_path)

    with open(yaml_file, "w") as file_handle:
        file_handle.write(yaml_example)

    process = subprocess.Popen(
        ["adca", str(yaml_file), "-hm", "1", "-pv", "1"],
        stdout=subprocess.PIPE,
    )
    _output, _error = process.communicate()
    assert process.returncode == 0  # Exit code 0 => everything OK


def test_CLI_init_then_run(tmp_path):
    os.chdir(tmp_path)
    process = subprocess.Popen(["adca", "init"], stdout=subprocess.PIPE)
    output, error = process.communicate()
    assert process.returncode == 0  # Exit code 0 => everything OK

    process = subprocess.Popen(
        ["adca", "run", "demo.yaml", "-hm", "1", "-pv", "0"],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    output, error = process.communicate()
    assert process.returncode == 0  # Exit code 0 => everything OK

    # Do a snapshot test on some lines
    # If this fails in the future then it is OK if the algorithms are changed,
    # but these results should not change unexpectedly. And should probably not change by much.
    lines = [line.strip() for line in output.split("\n")]
    assert (
        lines[82]
        == "Posterior theta (after pilot estimate): [14.84079942  4.03785857 -2.13500515]"
    )

    assert lines[85] == '"p": 1.5219168287045592,'
    assert lines[86] == '"sigma": 0.2053350412919966,'
    assert lines[87] == '"phi": 0.010199212150504245,'

    assert lines[136] == '"half_life": 8.806795605732418,'
    assert lines[137] == '"prior_strength": 0.0031622776601683794'

    assert lines[140] == "Negative log-likelihood: -0.3581"
    assert lines[141] == "RMSE in logspace: 0.2705"
    assert lines[142] == "Relative error (expected): 6.64%"
    assert lines[143] == "Relative error (P50): -15.33%"


if __name__ == "__main__":
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
