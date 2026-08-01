import numpy as np


def rotate(vector, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [vector[0] * c - vector[1] * s, vector[0] * s + vector[1] * c],
        dtype=np.float64,
    )


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
