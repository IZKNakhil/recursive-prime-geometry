import numpy as np

from .constants import COMPOSITE_COLORS


def add_arc(data, step, base_angles, composite_index, linewidth_prime, linewidth_composite):
    t = base_angles * step["signed_angle"]
    ct = np.cos(t)
    st = np.sin(t)
    center = step["center"]
    radial_start = step["radial_start"]
    x = center[0] + radial_start[0] * ct - radial_start[1] * st
    y = center[1] + radial_start[0] * st + radial_start[1] * ct
    arc = np.column_stack((x, y))

    if step["is_prime"]:
        color = "#000000"
        width = linewidth_prime
    else:
        color = COMPOSITE_COLORS[composite_index % len(COMPOSITE_COLORS)]
        width = linewidth_composite
        composite_index += 1

    data["arc_groups"].setdefault(color, {"segments": [], "linewidth": width})
    data["arc_groups"][color]["segments"].append(arc)
    return composite_index
