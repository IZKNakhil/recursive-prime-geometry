import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from .arcs import add_arc
from .arrays import create_arrays, store_step_arrays
from .constants import VALID_FAMILY_MODES, VALID_PRIME_DISPLAY_MODES
from .family import add_prime_family_point, calculate_family_centroids, family_key
from .geometry import build_geometry_step
from .plotting import draw_plot
from .sieve import sieve_eratosthenes
from .sim_statistics import build_statistics, create_statistics, update_statistics


def create_simulation_data():
    return {
        "prime_families": defaultdict(list),
        "all_families": defaultdict(list),
        "family_centers_raw": {},
        "family_normals": {},
        "family_arc_midpoints": {},
        "family_circle_centers": {},
        "family_line_points": defaultdict(list),
        "arc_groups": {},
        "prime_plot_points": [],
        "prime_plot_colors": [],
        "prime_plot_labels": [],
        "circle_centers": [],
        "arc_midpoints": [],
        "normal_segments": [],
    }


def validate_inputs(N, prime_display_mode, family_mode):
    if N < 2:
        raise ValueError("N muss mindestens 2 sein.")
    if prime_display_mode not in VALID_PRIME_DISPLAY_MODES:
        raise ValueError(
            "prime_display_mode muss 'points', 'labels', 'points+labels' oder 'none' sein."
        )
    if family_mode not in VALID_FAMILY_MODES:
        raise ValueError(
            "family_mode muss einer dieser Werte sein: "
            + ", ".join(sorted(VALID_FAMILY_MODES))
        )


def update_prime_history(is_prime, number, last_prime, gap_history):
    if not is_prime:
        return last_prime

    if last_prime is not None:
        gap_history.append(number - last_prime)

    return number


def remember_optional_points(data, step, show_circle_centers, show_arc_midpoints):
    if show_circle_centers:
        data["circle_centers"].append(step["center"].copy())
    if show_arc_midpoints:
        data["arc_midpoints"].append(step["midpoint"].copy())


def remember_local_normal(data, step, show_local_normals, family_spacing):
    if not show_local_normals:
        return

    data["normal_segments"].append(
        np.vstack(
            [
                step["midpoint"],
                step["midpoint"] + family_spacing * step["normal"],
            ]
        )
    )


def simulate_prime_geometry(
    N=1_000_000,
    radius=1.0,
    snapshot_values=None,
    snapshot_folder=None,
    rotation_angle=90.0,
    gap_depth=8,
    region_size=10_000_000,
    family_spacing=0.5,
    angle_tolerance=1e-6,
    distance_tolerance=1e-6,
    show_arcs=True,
    prime_display_mode="points",
    show_composite_arcs=False,
    show_family_lines=False,
    show_family_centers=False,
    show_centroid=False,
    show_start_point=True,
    show_end_point=True,
    show_circle_centers=False,
    show_arc_midpoints=False,
    show_local_normals=False,
    show_family_centroids=False,
    show_between_family_lines=False,
    arc_resolution=6,
    linewidth_prime=2.0,
    linewidth_composite=0.5,
    prime_point_size=10,
    family_center_size=22,
    centroid_size=90,
    figsize=(12, 12),
    dpi=150,
    save_path=None,
    return_arrays=False,
    max_labels=5000,
    family_mode="same_arc",
    state_depth=16,
    draw=True,
):
    validate_inputs(N, prime_display_mode, family_mode)

    if snapshot_values is None:
        snapshot_values = []

    snapshot_values = set(snapshot_values)

    is_prime_lookup = sieve_eratosthenes(N)
    rotation_rad = np.deg2rad(float(rotation_angle))
    phase_size = max(1, int(np.ceil((N + 1) / 10.0)))
    base_angles = np.linspace(0.0, 1.0, arc_resolution + 1)

    pos = np.array([0.0, 0.0], dtype=np.float64)
    heading = np.array([1.0, 0.0], dtype=np.float64)
    heading_angle = 0.0

    state_history = deque(maxlen=state_depth)
    gap_history = deque(maxlen=gap_depth)
    last_prime = None
    composite_index = 0

    data = create_simulation_data()
    stats = create_statistics(pos)
    arrays = create_arrays(N - 1) if return_arrays else None

    for idx, number in enumerate(range(2, N + 1)):

        if number % 100_000 == 0:
            progress = 100 * number / N
            print(f"{progress:5.1f}%   ({number:,} / {N:,})")

        if number in snapshot_values:
            print(f"[{number:,} / {N:,}] Snapshot wird gespeichert...")

        is_prime = bool(is_prime_lookup[number])

        last_prime = update_prime_history(is_prime, number, last_prime, gap_history)

        state_history.append(1 if is_prime else 0)

        step = build_geometry_step(
            number,
            is_prime,
            pos,
            heading,
            heading_angle,
            radius,
            rotation_rad,
        )

        update_statistics(stats, step, angle_tolerance)

        if arrays is not None:
            store_step_arrays(arrays, idx, step)

        if show_arcs and (is_prime or show_composite_arcs):
            composite_index = add_arc(
                data,
                step,
                base_angles,
                composite_index,
                linewidth_prime,
                linewidth_composite,
            )

        remember_optional_points(data, step, show_circle_centers, show_arc_midpoints)
        remember_local_normal(data, step, show_local_normals, family_spacing)

        key = family_key(
            family_mode,
            step,
            angle_tolerance,
            distance_tolerance,
            region_size,
            state_history,
            gap_history,
            state_depth,
            gap_depth,
        )

        if key is not None:
            data["all_families"][key].append(number)

        if is_prime and key is not None:
            add_prime_family_point(
                data,
                key,
                step,
                family_spacing,
                phase_size,
                prime_display_mode,
                show_family_lines,
            )

        pos = step["end_pos"]
        heading = step["end_heading"]
        heading_angle = step["end_heading_angle"]

        if snapshot_folder is not None and number in snapshot_values:

            family_centroids, _ = calculate_family_centroids(
                data,
                family_spacing,
            )

            centroid, statistics = build_statistics(
                number,  # <-- nicht N!
                radius,
                rotation_angle,
                pos,
                heading,
                data,
                stats,
            )

            fig, ax = draw_plot(
                data,
                centroid,
                family_centroids,
                pos,
                number,  # <-- nicht N!
                statistics["largest_family"],
                show_arcs,
                show_local_normals,
                show_circle_centers,
                show_arc_midpoints,
                prime_display_mode,
                prime_point_size,
                max_labels,
                show_family_lines,
                show_family_centers,
                family_center_size,
                show_family_centroids,
                show_between_family_lines,
                show_start_point,
                show_end_point,
                show_centroid,
                centroid_size,
                figsize,
                dpi,
                save_path=f"{snapshot_folder}/Primzahl_{number}.png",
            )

            plt.close(fig)

    family_centroids, _ = calculate_family_centroids(data, family_spacing)

    centroid, statistics = build_statistics(
        N,
        radius,
        rotation_angle,
        pos,
        heading,
        data,
        stats,
    )

    fig = None
    ax = None

    if draw:
        fig, ax = draw_plot(
            data,
            centroid,
            family_centroids,
            pos,
            N,
            statistics["largest_family"],
            show_arcs,
            show_local_normals,
            show_circle_centers,
            show_arc_midpoints,
            prime_display_mode,
            prime_point_size,
            max_labels,
            show_family_lines,
            show_family_centers,
            family_center_size,
            show_family_centroids,
            show_between_family_lines,
            show_start_point,
            show_end_point,
            show_centroid,
            centroid_size,
            figsize,
            dpi,
            save_path,
        )

    result = {
        "figure": fig,
        "axes": ax,
        "prime_families": data["prime_families"],
        "all_families": data["all_families"],
        "centroid": centroid,
        "family_centroids": family_centroids,
        "statistics": statistics,
    }

    if arrays is not None:
        result["arrays"] = arrays

    return result
