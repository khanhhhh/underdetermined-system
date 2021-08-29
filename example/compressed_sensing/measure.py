import numpy as np


def create_measure_matrix(D: int, N: int) -> np.ndarray:
    """
    create_sensing_matrix
    :param D: number of measure signal data points
    :param N: number of true signal data points
    :return: sensing matrix
    """
    M = np.zeros(shape=(D, N), dtype=np.int64)
    indices = np.random.choice(np.arange(N), size=D, replace=False)
    for d in range(D):
        M[d, indices[d]] = 1
    return M
