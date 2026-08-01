import numpy as np


def create_arrays(size):
    return {
        "n": np.empty(size, dtype=np.int64),
        "is_prime": np.empty(size, dtype=np.bool_),
        "start_points": np.empty((size, 2), dtype=np.float64),
        "end_points": np.empty((size, 2), dtype=np.float64),
        "circle_centers": np.empty((size, 2), dtype=np.float64),
        "arc_midpoints": np.empty((size, 2), dtype=np.float64),
        "tangents": np.empty((size, 2), dtype=np.float64),
        "normals": np.empty((size, 2), dtype=np.float64),
        "heading_angles": np.empty(size, dtype=np.float64),
        "turn": np.empty(size, dtype=np.int8),
    }


def store_step_arrays(arrays, idx, step):
    arrays["n"][idx] = step["number"]
    arrays["is_prime"][idx] = step["is_prime"]
    arrays["start_points"][idx] = step["start_pos"]
    arrays["end_points"][idx] = step["end_pos"]
    arrays["circle_centers"][idx] = step["center"]
    arrays["arc_midpoints"][idx] = step["midpoint"]
    arrays["tangents"][idx] = step["tangent"]
    arrays["normals"][idx] = step["normal"]
    arrays["heading_angles"][idx] = step["start_heading_angle"]
    arrays["turn"][idx] = step["turn_sign"]
