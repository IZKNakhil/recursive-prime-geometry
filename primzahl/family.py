import numpy as np
import matplotlib.cm as cm
from collections import Counter

from .constants import VALID_FAMILY_MODES
from .quantization import quantize_angle, quantize_value


def family_key(
    mode,
    step,
    angle_tolerance,
    distance_tolerance,
    region_size,
    state_history=None,
    gap_history=None,
    state_depth=None,
    gap_depth=None,
):
    if mode == "same_arc":
        return (
            quantize_value(step["center"][0], distance_tolerance),
            quantize_value(step["center"][1], distance_tolerance),
            quantize_angle(step["start_heading_angle"], angle_tolerance),
            int(step["turn_sign"]),
        )
    if mode == "resonance":
        return (
            quantize_angle(step["normal_angle"], angle_tolerance),
            quantize_angle(step["resonance_angle"], angle_tolerance),
            quantize_value(step["center"][0], distance_tolerance),
            quantize_value(step["center"][1], distance_tolerance),
        )
    if mode == "state_resonance":
        return tuple(state_history) if len(state_history) >= state_depth else None
    if mode == "gap_resonance":
        return tuple(gap_history) if len(gap_history) >= gap_depth else None
    if mode == "normal_family":
        return (quantize_angle(step["normal_angle"], angle_tolerance),)
    if mode == "normal_heading_family":
        return (
            quantize_angle(step["normal_angle"], angle_tolerance),
            quantize_angle(step["start_heading_angle"], angle_tolerance),
        )
    if mode == "circle_center":
        return (
            quantize_value(step["center"][0], distance_tolerance),
            quantize_value(step["center"][1], distance_tolerance),
        )
    if mode == "center_region":
        return (
            int(np.floor(step["center"][0] / region_size)),
            int(np.floor(step["center"][1] / region_size)),
        )
    if mode == "diagonal_resonance":
        return (
            quantize_value(step["center"][0] + step["center"][1], distance_tolerance),
        )

    raise ValueError(
        "family_mode muss einer dieser Werte sein: "
        + ", ".join(sorted(VALID_FAMILY_MODES))
    )


def add_prime_family_point(
    data,
    key,
    step,
    family_spacing,
    phase_size,
    prime_display_mode,
    show_family_lines,
):
    family_index = len(data["prime_families"][key])
    data["prime_families"][key].append(step["number"])

    if key not in data["family_centers_raw"]:
        data["family_centers_raw"][key] = step["midpoint"].copy()
        data["family_arc_midpoints"][key] = step["midpoint"].copy()
        data["family_circle_centers"][key] = step["center"].copy()
        data["family_normals"][key] = step["normal"].copy()

    arranged_point = (
        data["family_arc_midpoints"][key]
        + (family_index + 1) * family_spacing * data["family_normals"][key]
    )

    if prime_display_mode != "none":
        t = step["number"] / phase_size / 10.0
        t = np.clip(t, 0.0, 1.0)
        data["prime_plot_colors"].append(cm.turbo(t))

        data["prime_plot_points"].append(arranged_point)
        data["prime_plot_labels"].append(str(step["number"]))

    if show_family_lines:
        data["family_line_points"][key].append(arranged_point)


def calculate_family_centroids(data, family_spacing):
    family_centroids = {}
    family_sizes = {}

    for key, members in data["prime_families"].items():
        family_sizes[key] = len(members)
        base = data["family_arc_midpoints"][key]
        normal = data["family_normals"][key]
        offsets = np.arange(1, len(members) + 1, dtype=np.float64)[:, None]
        family_centroids[key] = (base + offsets * family_spacing * normal).mean(axis=0)

    return family_centroids, family_sizes


def summarize_family_sizes(prime_families):
    values = np.fromiter(
        (len(members) for members in prime_families.values()),
        dtype=np.int64,
        count=len(prime_families),
    )
    if len(values) == 0:
        return 0, 0.0, 0, {}

    return (
        int(values.max()),
        float(values.mean()),
        int(np.count_nonzero(values > 1)),
        dict(Counter(values.tolist())),
    )
