from unittest.mock import mock_open, patch
import os
import pytest
import sys
import numpy as np
import dis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.curve_fitting import round_to_len



@pytest.mark.parametrize("value,digits,expected_value", [
    (4.5, 4, 4.5),
    (0, 4, 0),
    (123.0, 4, np.int64(123)),
    (-100.0, 3, np.int64(-100)),
    (12345.6789, 4, 12350.0),
    (987654321, 5, np.int64(987650000)),
    (0.000987654, 2, 0.00099),
    (-0.00012345, 3, -0.000123),
    (np.inf, 4, np.inf),
    (-np.inf, 4, -np.inf),
    (np.nan, 4, np.nan),
])
def test_round_to_len(value, digits, expected_value):
    rounded_value = round_to_len(value, digits=digits)

    if np.isnan(expected_value):
        assert np.isnan(rounded_value)
    else:
        assert rounded_value == expected_value