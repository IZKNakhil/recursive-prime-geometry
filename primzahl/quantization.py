import numpy as np


def quantize_value(value, tolerance):
    if tolerance <= 0:
        return round(float(value), 12)
    return int(round(float(value) / tolerance))


def quantize_angle(angle, tolerance):
    angle = float(angle % (2.0 * np.pi))
    if tolerance <= 0:
        return round(angle, 12)
    return int(round(angle / tolerance))
