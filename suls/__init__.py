import time

import numpy as np
import cvxpy as cp


def check_arguments(a: np.ndarray, b: np.ndarray):
    if len(a.shape) != 2 or len(b.shape) != 1:
        raise Exception("A must be 2D, b must be 1D")

    m, n = a.shape

    if not (m < n):
        raise Exception("System must be under-determined (m < n)")


def solve_lp(A: np.ndarray, b: np.ndarray, p: int = 1) -> np.ndarray:
    """
    Minimizer of ||x||_p^p
    Given Ax=b
    By minimizing ||Ax-b||_2^2 + ||x||_p^p
    """

    check_arguments(A, b)
    m, n = A.shape

    t1 = time.time()
    x = cp.Variable(shape=(n,))
    objective = cp.Minimize(cp.sum_squares(A @ x - b) + cp.norm(x, p))
    problem = cp.Problem(objective=objective)
    t2 = time.time()
    print(f"model building... {t2 - t1}s")
    problem.solve()
    t3 = time.time()
    print(f"model solving.... {t3 - t2}s")
    return x.value


def solve_l1(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimizer of ||x||_1
    Given Ax=b

    Linear program
    Minimize 1^T y (sum of all elements of y)
    Given y >= |x| and Ax=b
    """
    check_arguments(A, b)
    m, n = A.shape
    t1 = time.time()
    x = cp.Variable(shape=(n,))
    y = cp.Variable(shape=(n,))
    problem = cp.Problem(
        objective=cp.Minimize(cp.sum(y)),
        constraints=[
            y >= x,
            y >= -x,
            A @ x == b,
        ]
    )
    t2 = time.time()
    print(f"model building... {t2 - t1}s")
    problem.solve()
    t3 = time.time()
    print(f"model solving.... {t3 - t2}s")
    return x.value


