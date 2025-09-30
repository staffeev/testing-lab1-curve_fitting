import os
import io
import pytest
import builtins
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.build_objectives import parse_formula, parse_source, FORMULA_START_TO_TYPE

### ------ Парсинг формул --------- ###


@pytest.mark.parametrize("formula,expected_vars", [
    ("x", 1),
    ("-x", 1),
    ("xy", 2),
    ("xxyzzzyxyzy", 3),
    ("aaa", 0),
    ("", 0),
])
def test_parse_formula_variables(formula, expected_vars):
    _, num_vars, _ = parse_formula(formula)
    assert num_vars == expected_vars


@pytest.mark.parametrize("formula,expected_parsed", [
    ("5", "(5)"),
    ("-2.5", "(-2.5)"),
    ("^-1.5", "**(-1.5)"),
    ("", ""),
    ("(3.5)", "((3.5))"),
])
def test_parse_formula_digits(formula, expected_parsed):
    parsed, _, _ = parse_formula(formula)
    assert parsed == expected_parsed


@pytest.mark.parametrize("formula,expected_parsed", [
    # ("e^x", "(np.e)**(x)"),  # TODO:
    ("e^1.5", "(np.e)**(1.5)"),
    # ("e^2", "(np.e)**(2)"),  # TODO:
    ("z^-2", "(z)**(-2)"),
    ("x^(1)y^2.5x^-2", "(x)**((1))*(y)**(2.5)*(x)**(-2)"),
    #("x^e", "(x)**(np.e)")  # TODO
])
def test_parse_formula_pow(formula, expected_parsed):
    parsed, _, _ = parse_formula(formula)
    assert parsed == expected_parsed


@pytest.mark.parametrize("formula,expected_parsed", [
    ("lnx", "(np.log(x))"),
    ("ln2.5", "np.log(2.5)"),
    ("xlny", "(x)*(np.log(y))"),
    ("lnxy", "(np.log(x))*(y)")
])
def test_parse_formula_log(formula, expected_parsed):
    parsed, _, _ = parse_formula(formula)
    assert parsed == expected_parsed


@pytest.mark.parametrize("formula,expected_parsed", [
    (")(", ")*("),
    ("((()))", "((()))"),
    (")()()()()", ")*()*()*()*()"),
    (")()(())", ")*()*(())"),
    ("(*", "*("),
    ("(*()*(", "*(()*(")
])
def test_parse_formula_braces(formula, expected_parsed):
    parsed, _, _ = parse_formula(formula)
    assert parsed == expected_parsed


@pytest.mark.parametrize("formula,expected_constants", [
    ("A", ["A"]),
    ("Ax+By", ["A", "B"]),
    ("CDZ", ["C", "D", "Z"]),
    ("aAbB", ["A", "B"]),
    ("", []),
    # ("A+A*A+AAA", ["A"])  # TODO
])
def test_parse_formula_constants(formula, expected_constants):
    _, _, constants = parse_formula(formula)
    assert constants == expected_constants


def test_parse_formula_variables():
    formula = "A+A*A+AAA"
    parsed, num_vars, constants = parse_formula(formula)
    print(parsed, num_vars, constants)

test_parse_formula_variables()

# # ---------- Тесты parse_formula ----------

# def test_parse_formula_simple_polynomial():
#     formula = "z=A+Bx^2+Cy"
#     parsed, num_vars, constants = parse_formula(formula)
#     assert "xy[0]" in parsed
#     assert "xy[1]" in parsed
#     assert num_vars == 3
#     assert constants == ["A", "B", "C"]


# def test_parse_formula_with_log():
#     formula = "y=A+ln(x)"
#     parsed, num_vars, constants = parse_formula(formula)
#     assert "np.log" in parsed  # заменилось на numpy.log
#     assert num_vars == 2
#     assert "A" in constants


# def test_parse_formula_with_exp():
#     formula = "y=A+e^x"
#     parsed, num_vars, constants = parse_formula(formula)
#     assert "(np.e)" in parsed
#     assert num_vars == 2
#     assert "A" in constants


# def test_parse_formula_with_fractional_power():
#     formula = "y=A+Bx^(0.5)"
#     parsed, num_vars, constants = parse_formula(formula)
#     assert "**((0.5))" in parsed  # степень заменена на **()
#     assert num_vars == 2
#     assert "B" in constants


# def test_parse_formula_with_inverse_power():
#     formula = "y=A+C/x"
#     parsed, num_vars, constants = parse_formula(formula)
#     assert "/(x)" in parsed or "/x" in parsed  # проверяем, что инверсия сохранилась
#     assert num_vars == 2
#     assert "C" in constants


# @pytest.mark.parametrize("formula,expected_type", [
#     ("y=A+x", "simple"),
#     ("y=A+x**2", "square"),
#     ("y=np.log(x)", "ln"),
#     ("y=x**((0.5))", "sqrt"),
#     ("y=x**((-1))", "inv"),
#     ("y=(e)**(x)", "exp"),
# ])
# def test_formula_type_map(formula, expected_type):
#     # проверим, что FORMULA_START_TO_TYPE умеет находить тип по left части
#     left = formula.split("=")[0].replace("y", "").replace("z", "")
#     assert isinstance(expected_type, str)
#     # тут мы только убеждаемся, что тип есть в словаре
#     assert expected_type in FORMULA_START_TO_TYPE.values()


# def test_parse_formula_empty_string():
#     with pytest.raises(Exception):
#         parse_formula("")


# def test_parse_formula_without_variables():
#     formula = "y=5"
#     parsed, num_vars, constants = parse_formula(formula)
#     assert num_vars == 1  # только y
#     assert constants == []


# def test_parse_formula_invalid_syntax():
#     with pytest.raises(Exception):
#         parse_formula("y=A+@x")


# # ---------- Тесты parse_source ----------

# def test_parse_source_generates_file(tmp_path):
#     formulas = ["z=A+Bx^2+Cy", "y=A+ln(x)"]
#     test_file = tmp_path / "objectives.py"

#     # подменяем путь OBJECTIVES_PATH динамически
#     from source import build_objectives
#     build_objectives.OBJECTIVES_PATH = str(test_file)

#     parse_source(formulas)

#     content = test_file.read_text(encoding="utf-8")

#     # Проверяем, что нужные куски попали в файл
#     assert "import numpy as np" in content
#     assert "'form': 'z=A+Bx^2+Cy'" in content
#     assert "lambda xy, A, B, C" in content
#     assert "lambda x, A" in content or "lambda x, A" in content


# def test_parse_source_handles_multiple_types(tmp_path):
#     formulas = [
#         "y=A+Bx",           # simple
#         "y=A+Bx^2",         # square
#         "y=A+ln(x)",        # ln
#         "y=A+e^x",          # exp
#         "y=A+Bx^(0.5)",     # sqrt
#         "y=A+C/x",          # inv
#     ]
#     test_file = tmp_path / "objectives.py"

#     from source import build_objectives
#     build_objectives.OBJECTIVES_PATH = str(test_file)

#     parse_source(formulas)

#     content = test_file.read_text(encoding="utf-8")
#     for form in formulas:
#         assert f"'form': '{form}'" in content
