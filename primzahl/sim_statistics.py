import numpy as np

from .family import summarize_family_sizes
from .quantization import quantize_angle


def create_statistics(pos):
    return {
        "all_midpoint_sum": np.array([0.0, 0.0], dtype=np.float64),
        "prime_midpoint_sum": np.array([0.0, 0.0], dtype=np.float64),
        "composite_midpoint_sum": np.array([0.0, 0.0], dtype=np.float64),
        "path_position_sum": pos.copy(),
        "path_position_count": 1,
        "prime_count": 0,
        "composite_count": 0,
        "angle_counter": {},
        "normal_counter": {},
        "resonance_counter": {},
    }


def increase_counter(counter, key):
    counter[key] = counter.get(key, 0) + 1


def update_statistics(stats, step, angle_tolerance):
    stats["all_midpoint_sum"] += step["midpoint"]
    stats["path_position_sum"] += step["end_pos"]
    stats["path_position_count"] += 1

    increase_counter(
        stats["angle_counter"],
        quantize_angle(step["start_heading_angle"], angle_tolerance),
    )
    increase_counter(
        stats["normal_counter"],
        quantize_angle(step["normal_angle"], angle_tolerance),
    )
    increase_counter(
        stats["resonance_counter"],
        quantize_angle(step["resonance_angle"], angle_tolerance),
    )

    if step["is_prime"]:
        stats["prime_count"] += 1
        stats["prime_midpoint_sum"] += step["midpoint"]
    else:
        stats["composite_count"] += 1
        stats["composite_midpoint_sum"] += step["midpoint"]


def build_statistics(N, radius, rotation_angle, pos, heading, data, stats):
    largest_family, mean_family_size, repeated_families, distribution = summarize_family_sizes(
        data["prime_families"]
    )
    centroid = stats["all_midpoint_sum"] / max(1, N - 1)
    path_centroid = stats["path_position_sum"] / max(1, stats["path_position_count"])

    prime_centroid = np.array([np.nan, np.nan])
    if stats["prime_count"] > 0:
        prime_centroid = stats["prime_midpoint_sum"] / stats["prime_count"]

    composite_centroid = np.array([np.nan, np.nan])
    if stats["composite_count"] > 0:
        composite_centroid = stats["composite_midpoint_sum"] / stats["composite_count"]

    return centroid, {
        "N": int(N),
        "radius": float(radius),
        "rotation_angle_degrees": float(rotation_angle),
        "prime_count": int(stats["prime_count"]),
        "composite_count": int(stats["composite_count"]),
        "family_count": int(len(data["prime_families"])),
        "repeated_family_count": int(repeated_families),
        "largest_family": int(largest_family),
        "mean_family_size": float(mean_family_size),
        "family_size_distribution": distribution,
        "centroid_all_arc_midpoints": centroid,
        "centroid_path_positions": path_centroid,
        "centroid_prime_arc_midpoints": prime_centroid,
        "centroid_composite_arc_midpoints": composite_centroid,
        "final_position": pos.copy(),
        "final_heading": heading.copy(),
        "angle_distribution": dict(stats["angle_counter"]),
        "normal_distribution": dict(stats["normal_counter"]),
        "resonance_distribution": dict(stats["resonance_counter"]),
    }
