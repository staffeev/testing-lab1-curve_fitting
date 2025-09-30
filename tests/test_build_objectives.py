from unittest.mock import mock_open, patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.build_objectives import parse_source


def test_parse_source_mock_open():
    formulas = [
        # 3D
        "z^(-1)=A+Bx^2lnx+Cylny",
        "lnz=A+Bx^(2.5)+Cy/lny",
        "z=(A+Blnx+C(lnx)^2+Dy+Ey^2)/(1+Flnx+Gy)",
        # 2D
        "lny=A+B/x^(1.5)+Clnx/x^2",
        "y^(-1)=A+B/x^(0.5)+Clnx/x^2",
        "y^2=A+Bx^(0.5)lnx",
        "y^(0.5)=A+Bx+Cx^2+Dx^3",
        "y=A+Bx+Cx^(0.5)lnx+Dx^(0.5)"
    ]

    expected_objectives = [
        # 3D
        "{'form': 'z^(-1)=A+Bx^2lnx+Cylny', 'fun': lambda xy, A, B, C: A+B*(xy[0])**(2)*(np.log(xy[0]))+C*(xy[1])*(np.log(xy[1])), 'typ': 'inv', 'dim': 3, 'params': ('A', 'B', 'C')},",
        "{'form': 'lnz=A+Bx^(2.5)+Cy/lny', 'fun': lambda xy, A, B, C: A+B*(xy[0])**((2.5))+C*(xy[1])/(np.log(xy[1])), 'typ': 'ln', 'dim': 3, 'params': ('A', 'B', 'C')},",
        "{'form': 'z=(A+Blnx+C(lnx)^2+Dy+Ey^2)/(1+Flnx+Gy)', 'fun': lambda xy, A, B, C, D, E, F, G: (A+B*(np.log(xy[0]))+C*((np.log(xy[0])))**(2)+D*(xy[1])+E*(xy[1])**(2))/((1)+F*(np.log(xy[0]))+G*(xy[1])), 'typ': 'simple', 'dim': 3, 'params': ('A', 'B', 'C', 'D', 'E', 'F', 'G')},",
        # 2D
        "{'form': 'lny=A+B/x^(1.5)+Clnx/x^2', 'fun': lambda x, A, B, C: A+B/(x)**((1.5))+C*(np.log(x))/(x)**(2), 'typ': 'ln', 'dim': 2, 'params': ('A', 'B', 'C')},",
        "{'form': 'y^(-1)=A+B/x^(0.5)+Clnx/x^2', 'fun': lambda x, A, B, C: A+B/(x)**((0.5))+C*(np.log(x))/(x)**(2), 'typ': 'inv', 'dim': 2, 'params': ('A', 'B', 'C')},",
        "{'form': 'y^2=A+Bx^(0.5)lnx', 'fun': lambda x, A, B: A+B*(x)**((0.5))*(np.log(x)), 'typ': 'square', 'dim': 2, 'params': ('A', 'B')},",
        "{'form': 'y^(0.5)=A+Bx+Cx^2+Dx^3', 'fun': lambda x, A, B, C, D: A+B*(x)+C*(x)**(2)+D*(x)**(3), 'typ': 'sqrt', 'dim': 2, 'params': ('A', 'B', 'C', 'D')},",
        "{'form': 'y=A+Bx+Cx^(0.5)lnx+Dx^(0.5)', 'fun': lambda x, A, B, C, D: A+B*(x)+C*(x)**((0.5))*(np.log(x))+D*(x)**((0.5)), 'typ': 'simple', 'dim': 2, 'params': ('A', 'B', 'C', 'D')},",
    ]

    m = mock_open()
    with patch("builtins.open", m):
        parse_source(formulas)

    m.assert_called_once()
    handle = m()
    for expected_objective, objective in zip(expected_objectives, handle.write.call_args_list[2:]):
        # print(expected_objective)
        # print(objective.args[0].strip())
        # print("-----------------------------")
        assert expected_objective == objective.args[0].strip()