import numpy as np


def forward(H1: int, W1: int, H2: int, W2: int) -> np.ndarray:
    M = np.empty(shape=(H2 * W2, H1 * W1), dtype=np.complex128)
    for h2 in range(H2):
        for w2 in range(W2):
            i2 = h2 * W2 + w2  # row major
            for h1 in range(H1):
                for w1 in range(W1):
                    i1 = h1 * W1 + w1  # row major
                    M[i2, i1] = complex(0, - 2 * np.pi * (h2 * h1 / H1 + w2 * w1 / W1))
    return np.exp(M) / (H1 * W1)


def backward(H2: int, W2: int, H1: int, W1: int) -> np.ndarray:
    M = np.empty(shape=(H1 * W1, H2 * W2), dtype=np.complex128)
    for h1 in range(H1):
        for w1 in range(W1):
            i1 = h1 * W1 + w1  # row major
            for h2 in range(H2):
                for w2 in range(W2):
                    i2 = h2 * W2 + w2  # row major
                    M[i1, i2] = complex(0, 2 * np.pi * (h2 * h1 / H1 + w2 * w1 / W1))
    return np.exp(M)
