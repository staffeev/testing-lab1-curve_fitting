from collections import defaultdict
from operator import itemgetter
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray
import numpy as np
from source.build_objectives import parse_formula, FORMULA_START_TO_TYPE
from source.curve_fitting import \
    find_fit_and_metrics, replace_constant_placeholders_with_numbers, TRANSFORMATION_FUNC, FUNC_TYPE_INVERSION


EQS = {
    r"([0-9])([(,x,y,z,l])": r"\1*\2",        # замена слитного написания типа ax на a*x, включая ln
    r"\^": r"**",                           # замена символа возведения в степень
    r"lnx": r"np.log(x)",                   # ln на np.log
    r"lny": r"np.log(y)",
    r"xnp": r"x*np",                        # xnp на x*np
    r"ynp": r"y*np",
    r"e\*\*": r"*np.e**",                   # e** на *np.e**
    r"\)np": r")*np"                        # )np на )*np
}


def get_solutions(
        context: Dict,
        x_list: List[float],
        y_list: List[float],
        z_list: Optional[List[float]] = None,
        rows_count: Optional[int] = None,
        max_parameters: Optional[int] = None,
        use_only_max_dimension: Optional[bool] = False) -> List[Dict]:
    """
    Obtain the table with the best solutions by 2d equation solver for points data

    :param context: Celery context
    :param x_list: first log data list (x-curve)
    :param y_list: second log data list (y-curve)
    :param z_list: third log data list (z-curve)
    :param rows_count: result table rows count per function type
    :param max_parameters: max params number of the analyzed functions ("params" field in objectives.py)
    :param use_only_max_dimension: if True and z_list is not None, solutions will be found only for 3D 
    :return: table with the best solutions by 2d solver
    """
    x_data = np.array(x_list)
    y_data = np.array(y_list)

    if z_list is not None: # find for 3D also
        z_data = np.array(z_list)
        fr_fit_metrics = find_fit_and_metrics(np.vstack((x_data, y_data)), z_data, max_parameters, 3)
        if not use_only_max_dimension:
            fr_fit_metrics.extend(find_fit_and_metrics(x_data, y_data, max_parameters, 2))
    else:
        fr_fit_metrics = find_fit_and_metrics(x_data, y_data, max_parameters, 2)

    results = []

    for fun_record, fit, metrics in fr_fit_metrics:
        formula = fun_record['form']
        params = fun_record['params']
        ndims = len(set(map(lambda x: x[0], re.findall(r"[x-z]", formula))))
        fin_eq = replace_constant_placeholders_with_numbers(formula, params, fit)

        results.append(
            (formula, fin_eq, len(params), *metrics, fun_record['typ'], ndims)
        )

    # make results table
    columns = ("formula", "formula_with_coefficients", "params_num", "r2",
               "adj_r2", "std_err", "max_err", "f_stat", "func_type", "n_dims")
    results.sort(key=itemgetter(8, 3), reverse=True)  # by type and then by r2

    sorted_table = [dict(zip(columns, res)) for res in results]

    # save results
    pd.DataFrame(sorted_table).to_csv("RESULT.csv", index=False)

    # we need rows_count best results of every function type we get (5 simple, 5 sqrt, etc)
    res_table = []

    type_cnt = defaultdict(int)
    for row in sorted_table:
        ft = row['func_type']
        ndims = row["n_dims"]
        cnt = type_cnt[(ndims, ft)]

        if rows_count is None or cnt < rows_count:
            res_table.append(row)
            type_cnt[(ndims, ft)] += 1

    return res_table


def evaluate_solver_equation(context: Dict, equation: str, x_data: NDArray,
                             y_data: Optional[NDArray] = None) -> List[Tuple[float, float]]:
    """
    Evaluate solver equation

    :param context: Celery context
    :param equation: equation as a formula string (2d solver result)
    :param x_data: x-values to evaluate formula
    :param y_data: y-values to evaluate formula (for 3D curves)
    :return: list of tuples (x:y) or ((x, y):z)
    """
    left, right = equation.split('=')
    num_variables = len(set(map(lambda x: x[0], re.findall(r"[x-z]", equation))))

    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float) if y_data is not None else None
    vals = np.array([x_data]).T if num_variables == 2 else np.vstack((x_data, y_data)).T

    # check equation for power functions
    power_pattern = r'y\^\(?(-?\d+\.?\d*)\)?' if num_variables == 2 else r'z\^\(?(-?\d+\.?\d*)\)?'
    match = re.fullmatch(power_pattern, left)
    additional_power = float(match.group(1)) if match else 1

    # check for exponent
    exp_res = left in ("lny", "lnz")

    # go through regex rules
    for exp, rep in EQS.items():
        right = re.sub(exp, rep, right)

    # evaluate
    try:
        with np.errstate(all='ignore'):
            eval_res = eval(right, {"np": np, "x": x_data, "y": y_data})
    except Exception:
        return []

    res = []
    vals = np.array([x_data]).T if num_variables == 2 else np.vstack((x_data, y_data)).T

    for er, val in zip(eval_res, vals):
        value = np.nan

        # if er is not number or infinite
        if er is None or not np.isfinite(er):
            res.append((*val, np.nan))
            continue

        try:
            # exp/ln
            if exp_res:
                value = np.exp(er)
            # inv
            elif additional_power < 0:
                if er != 0:
                    value = 1.0 / er
            # sqrt
            elif np.isclose(additional_power, 0.5):
                if er >= 0:
                    value = er ** 2
            # square
            elif np.isclose(additional_power, 2):
                if er >= 0:
                    value = np.sqrt(er)
            # other pow
            else:
                if er >= 0 or not additional_power % 1 == 0:
                    value = er ** (1 / additional_power)
        except Exception:
            value = np.nan

        res.append((*val, value))
    
    return res


