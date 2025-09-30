from re import findall, sub
import os
from typing import Iterable, List, Tuple


OBJECTIVES_PATH = os.path.join(os.path.dirname(__file__), 'objectives.py')

FUNCTION_PATTERN = "    {{'form': '{}', 'fun': lambda {}, {}: {}, 'typ': '{}', 'dim': {}, 'params': ({})}},\n"

REPLACE_EXP = r"(\g<0>)"   # обернуть выражение в скобки

RULES = {
    r"-?[xyz]": REPLACE_EXP,                  # переменная
    r"-?([0-9]+)\.?([0-9]*)": REPLACE_EXP,    # конкретное число
    r"[eyxz]\^\(+-?\w+\)+": REPLACE_EXP,      # переменная в числовой степени или exp
    r"e\^": r"(np.e)^",                       # е в степени
    r"ln\(\w+\)": REPLACE_EXP,                # натуральный логарифм
    r"[+-=*/\^][A-Z]\(": r"\g<0>*",           # коэффициент в уравнении
    r"\(\*": r"*(",                           # изменение (* на *(
    r"ln": r"np.log",                         # натуральный логарифм заменяется на log, чтобы numpy мог посчитать
    r"\)\(": r")*(",                          # умножение после степени
    r"\^": r"**"                              # знак степени
}

FORMULA_START_TO_TYPE = {
    r"()": "simple",
    r"()**((0.5))": "sqrt",
    r"()**(2)": "square",
    r"(np.log())": "ln",
    r"()**((-1))": "inv",
    r"((e)**())": "exp"
}


def parse_formula(formula: str) -> Tuple[str, List[str], List[str]]:
    """
    Parse formula from string to python func
    :param formula: formula string
    :return: formula string with its number of variables and constant placeholders
    """
    for exp, exp_to_replace in RULES.items():
        formula = sub(exp, exp_to_replace, formula)
    num_variables = len(set(map(lambda x: x[0], findall(r"[x-z]", formula))))
    constants = list(map(lambda x: x[0], findall(r"[A-Z][+*/\^]", formula)))
    if num_variables == 3:
        formula = sub(r"X", r"xy[0]", sub(r"y", r"xy[1]", sub(r"x", r"X", formula)))
    return formula, num_variables, constants


def parse_source(formulas: Iterable[str]):
    """
    Parse formulas from the list to the file `objectives.py`,
    get every function parameter and type rely on its Y transform

    :param formulas: formula list
    """
    with open(OBJECTIVES_PATH, 'w', encoding='utf-8') as res_file:

        res_file.write('import numpy as np\n\n\n')
        res_file.write('FUNCTIONS = (\n')

        for form_raw in formulas:
            form, variables, letters = parse_formula(form_raw)
            left, right = form.split("=")
            left = left.replace("y", "").replace("z", "")
            fty = FORMULA_START_TO_TYPE[left]
            args = ", ".join(letters)
            letters = ', '.join(f"'{sym}'" for sym in letters)

            var_symbol = "x" if variables == 2 else "xy"
            res_file.write(FUNCTION_PATTERN.format(form_raw, var_symbol, args, right, fty, variables, letters))

        res_file.write(')\n')


# Run this to update objectives.py file
if __name__ == '__main__':
    import pandas as pd
    filename = os.path.join(os.path.dirname(__file__), 'formulas.csv')
    parse_source(pd.read_csv(filename)["formula"].tolist())
