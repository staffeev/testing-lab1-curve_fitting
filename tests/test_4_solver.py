import os
import pytest
import sys
import numpy as np
import numpy.testing as npt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.solver import evaluate_solver_equation


@pytest.mark.parametrize(
    "equation,x,y,expected",
    [
        # 3д
        ("z^(-1)=1+2*x**2*np.log(x)+3*y*np.log(y)",
         np.array([1,2,3]), np.array([1,2,3]),
         [(1,1,1/(1+2*1**2*np.log(1)+3*1*np.log(1))),
          (2,2,1/(1+2*2**2*np.log(2)+3*2*np.log(2))),
          (3,3,1/(1+2*3**2*np.log(3)+3*3*np.log(3)))]
        ),

        ("lnz=1+2*x**2.5+3*y/np.log(y)",
         np.array([1,2,3]), np.array([2,3,4]),
         [(1,2,np.exp(1+2*1**2.5+3*2/np.log(2))),
          (2,3,np.exp(1+2*2**2.5+3*3/np.log(3))),
          (3,4,np.exp(1+2*3**2.5+3*4/np.log(4)))]
        ),

        ("z=(1+2*np.log(x)+3*(np.log(x))**2+4*y+5*y**2)/(1+6*np.log(x)+7*y)",
         np.array([1,2]), np.array([1,2]),
         [(1,1,(1+2*np.log(1)+3*np.log(1)**2+4*1+5*1**2)/(1+6*np.log(1)+7*1)),
          (2,2,(1+2*np.log(2)+3*np.log(2)**2+4*2+5*2**2)/(1+6*np.log(2)+7*2))]
        ),

        # 2д
        ("lny=1+2/x**1.5+3*np.log(x)/x**2",
         np.array([1,2,4]), None,
         [(1,np.exp(1+2/1**1.5+3*np.log(1)/1**2)),
          (2,np.exp(1+2/2**1.5+3*np.log(2)/2**2)),
          (4,np.exp(1+2/4**1.5+3*np.log(4)/4**2))]
        ),

        ("y^(-1)=1+2/x**0.5+3*np.log(x)/x**2",
         np.array([1,2,4]), None,
         [(1,1/(1+2/1**0.5+3*np.log(1)/1**2)),
          (2,1/(1+2/2**0.5+3*np.log(2)/2**2)),
          (4,1/(1+2/4**0.5+3*np.log(4)/4**2))]
        ),

        ("y^2=1+2*x**0.5*np.log(x)",
         np.array([1,4]), None,
         [(1,np.sqrt(1+2*1**0.5*np.log(1))),
          (4,np.sqrt(1+2*4**0.5*np.log(4)))]
        ),

        ("y^(0.5)=1+2*x+3*x**2+4*x**3",
         np.array([0.5,1]), None,
         [(0.5,(1+2*0.5+3*0.5**2+4*0.5**3)**2),
          (1,(1+2*1+3*1**2+4*1**3)**2)]
        ),

        ("y=1+2*x+3*x**0.5*np.log(x)+4*x**0.5",
         np.array([1,4]), None,
         [(1,1+2*1+3*1**0.5*np.log(1)+4*1**0.5),
          (4,1+2*4+3*4**0.5*np.log(4)+4*4**0.5)]
        ),

        # граничные случаи
        ("lny=1+np.log(x)",
         np.array([-1,0,1,2]), None,
         [(-1, np.nan),
          (0, np.nan),
          (1, np.exp(1+np.log(1))),
          (2, np.exp(1+np.log(2)))]
        ),

        ("y^2=-1+2*x",
         np.array([0,0.5,1]), None,
         [(0, np.nan),
          (0.5, np.sqrt(-1+2*0.5)),
          (1, np.sqrt(-1+2*1))]
        ),

        ("y^(0.5)=x-1",
         np.array([0,0.5,1,2]), None,
         [(0, np.nan),
          (0.5, np.nan),
          (1, np.sqrt(1-1)),
          (2, np.sqrt(2-1))]
        ),

        ("y^(-1)=1/x",
         np.array([-1,0,1,2]), None,
         [(-1, 1/(-1)),
          (0, np.nan),
          (1, 1),
          (2, 2)]
        ),
    ]
)
def test_evaluate_solver_equation(equation, x, y, expected):
    res = evaluate_solver_equation({}, equation, x, y)
    npt.assert_allclose(res, expected, atol=1e-6, equal_nan=True)
