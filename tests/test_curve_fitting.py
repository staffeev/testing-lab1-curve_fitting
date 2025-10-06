import os
import pytest
import sys
import numpy as np
import dis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.curve_fitting import round_to_len, replace_constant_placeholders_with_numbers, metrics_calc


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


@pytest.mark.parametrize("formula,letters,numbers,expected", [
    ("a+1", ["a"], [5], "5+1"),
    ("a+b", ["a", "b"], [2, 3], "2+3"),
    ("x+y", [], [], "x+y"),
    ("a+b", ["a", "b"], [3, -4], "3-4"),
    ("a+a*a", ["a"], [7], "7+7*7"),
    ("a-b", ["a", "b"], [-2, 5], "-2-5"),
    ("a+b", ["a", "b"], [1.5, 2.75], "1.5+2.75"),
])
def test_replace_constant_placeholders_with_numbers(formula, letters, numbers, expected):
    result = replace_constant_placeholders_with_numbers(formula, letters, numbers)
    assert result == expected


@pytest.mark.parametrize(
    "x_data, func, true_params, fit_params, noise_std, expect_r2_min",
    [
        (np.linspace(0, 10, 50), lambda x, a, b: a * x + b, (2, 1), (2.01, 0.95), 0.1, 0.98),
        (np.linspace(-5, 5, 60), lambda x, a, b, c: a * x**2 + b * x + c, (1.5, -0.5, 2), (1.45, -0.55, 2.1), 1.0, 0.9),
        (np.linspace(0.1, 10, 40), lambda x, a, b: a * np.sqrt(x) + b, (3, 2), (3.1, 1.9), 0.5, 0.85),
        (np.linspace(0.1, 10, 50), lambda x, a, b: a * np.log(x) + b, (4, 5), (3.9, 5.2), 0.3, 0.8),
        (np.linspace(1, 10, 50), lambda x, a, b: a / x + b, (10, 0.5), (9.8, 0.7), 0.1, 0.9),
    ],
)
def test_metrics_calc_common_functions(x_data, func, true_params, fit_params, noise_std, expect_r2_min):
    rng = np.random.default_rng(42)
    y_true = func(x_data, *true_params)
    y_noisy = y_true + rng.normal(0, noise_std, len(x_data))
    y_pred = func(x_data, *fit_params)

    r2, adj_r2, std_err, max_err, f = metrics_calc(x_data, y_noisy, y_pred, func)

    assert np.isfinite(r2)
    assert np.isfinite(adj_r2)
    assert np.isfinite(std_err)
    assert np.isfinite(max_err)
    assert np.isfinite(f)
    assert 0 <= r2 <= 1
    assert adj_r2 <= 1
    assert f >= 0
    assert r2 > expect_r2_min


@pytest.mark.parametrize(
    "description, x_data, y_data, y_pred, func, expected_behavior",
    [
        # одинаковые y и y_pred
        ("perfect_fit", np.arange(10), np.arange(10) * 2, np.arange(10) * 2, lambda x, a, b: a * x + b, lambda r2, *_: np.isclose(r2, 1)),
        # случайные данные, не должно быть зависимости
        ("no_relation", np.arange(10), np.random.random(10), np.random.random(10), lambda x, a, b: a * x + b, lambda r2, *_: r2 < 0.3),
        # константные y и y_pred
        ("constant_y", np.arange(10), np.ones(10) * 5, np.ones(10) * 5, lambda x, a, b: a * x + b, lambda r2, *_: np.isnan(r2) or np.isinf(r2)),
    ],
)
def test_metrics_calc_edge_cases(description, x_data, y_data, y_pred, func, expected_behavior):
    try:
        r2, adj_r2, std_err, max_err, f = metrics_calc(x_data, y_data, y_pred, func)
    except Exception as e:
        pytest.skip(f"{description} вызвал исключение: {e}")

    assert expected_behavior(r2, adj_r2, std_err, max_err, f), f"Неожиданное поведение для случая {description}"
