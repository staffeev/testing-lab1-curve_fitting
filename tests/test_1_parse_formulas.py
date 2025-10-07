import os
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.build_objectives import parse_formula


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
    ("e^x", "((np.e)**(x))"),
    ("e^1.5", "((np.e)**(1.5))"),
    ("e^2", "((np.e)**(2))"),
    ("z^-2", "((z)**(-2))"),
    ("x^(1)y^2.5x^-2", "(x)**((1))*(y)**(2.5)*(x)**(-2)"),
    ("x^e", "((x)**(np.e))")
])
def test_parse_formula_pow(formula, expected_parsed):
    parsed, _, _ = parse_formula(formula)
    print(parsed)
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