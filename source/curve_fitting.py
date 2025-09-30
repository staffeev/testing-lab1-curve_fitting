from typing import Any, List, Callable, Optional, Tuple, Union
import warnings
from inspect import signature

from scipy.optimize import curve_fit
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import max_error

from source.objectives import FUNCTIONS


TRANSFORMATION_FUNC = {
    "simple": np.ravel,
    "square": np.square,
    "sqrt": np.sqrt,
    "inv": np.reciprocal,
    "ln": np.log,
    "exp": np.exp
}
FUNC_TYPE_INVERSION = {
    "simple": "simple",
    "sqrt": "square",
    "square": "sqrt",
    "inv": "inv",
    "ln": "exp",
    "exp": "ln"
}


def round_to_len(value: np.floating[Any], digits: int = 4) -> Union[np.floating[Any], np.integer[Any]]:
    """
    Round value to significant precision

    :param value: value to round
    :param digits: number of significant digits
    :return: rounded value
    """
    if value == 0 or not np.isfinite(value):
        return value

    left_digits = int(np.floor(np.log10(np.abs(value)))) + 1
    precision = max(0, digits - left_digits)

    rv = np.round(value, precision)

    if np.modf(rv)[0] == 0:  # no fractional part
        return np.int64(rv)
    else:
        return rv


def metrics_calc(x_data: NDArray, y_data: NDArray, y_predicted: NDArray, func: Callable) -> Tuple:
    """
    Calculate function metrics

    :param x_data: x-values
    :param y_data: y-values
    :param y_predicted: y predicted data
    :param func: function
    :return: calculated metrics
    """
    residuals = y_data - y_predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(x_data) if x_data.ndim == 1 else len(x_data[0])
    k = len(signature(func).parameters) - 1
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    std_d_err = np.sqrt(np.sum((y_data - np.mean(y_data)) ** 2) / (n - 1))
    std_err = np.sqrt(1 - adj_r2) * std_d_err
    max_err = max_error(y_data, y_predicted)
    f = (r2 / (1 - r2)) * ((n - k - 1) / k)

    return r2, adj_r2, std_err, max_err, f


def process_function(func: Callable, x_data: NDArray, y_data: NDArray, func_type: str) -> Optional[Tuple[List, List]]:
    """
    Process given function via its type to lambda convert

    :param func: function
    :param x_data: x-values
    :param y_data: y-values
    :param func_type: function type string
    :return: fit and metrics values
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            fit = curve_fit(func, x_data, TRANSFORMATION_FUNC[func_type](y_data), maxfev=100000)[0]
            fit_rd = [round_to_len(val, digits=5) for val in fit]

            predicted_data = TRANSFORMATION_FUNC[FUNC_TYPE_INVERSION[func_type]](func(x_data, *fit_rd))

            metrics = metrics_calc(x_data, y_data, predicted_data, func)
            metrics_rd = [round_to_len(val, digits=5) for val in metrics]

            return fit_rd, metrics_rd

    except (RuntimeError, ValueError):
        return None


def find_fit_and_metrics(x_data: NDArray, y_data: NDArray, max_parameters: Optional[int], 
                         dim: int = 2) -> List:
    """
    Process given function (use non-linear least squares to fit a function and calculate its metrics)

    :param x_data: x-values
    :param y_data: y-values
    :param max_parameters: max params number of the analyzed functions
    :param dim: number of curve dimensions (2 or 3)
    :return: list of function record, function fitted and its metrics
    """
    results = []
    if max_parameters is None:
        max_parameters = len(x_data) if x_data.ndim == 1 else len(x_data[0])
    else: # curve_fit limitation
        max_parameters = min(max_parameters, len(x_data) if x_data.ndim == 1 else len(x_data[0])) 
    for fun_record in FUNCTIONS:
        if len(fun_record['params']) > max_parameters or fun_record["dim"] != dim:
            continue
        processing_result = process_function(fun_record['fun'], x_data, y_data, fun_record['typ'])

        if processing_result:
            fit, metrics = processing_result
            results.append((fun_record, fit, metrics))

    return results


def replace_constant_placeholders_with_numbers(formula: str, letters: List[str], numbers: List[float]) -> str:
    """
    Replace chars coeffs in formula with numbers

    :param formula: formula string
    :param letters: coeffs letters
    :param numbers: calculated numbers values
    :return: equation
    """
    pattern = dict(zip(letters, list(map(str, numbers))))
    new_f = formula.translate(formula.maketrans(pattern))
    new_f = new_f.replace("+-", "-")

    return new_f
