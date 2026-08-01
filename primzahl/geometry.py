import numpy as np

from .vector_math import normalize, rotate


def build_geometry_step(number, is_prime, pos, heading, heading_angle, radius, rotation_rad):
    turn_sign = 1 if is_prime else -1
    signed_angle = turn_sign * rotation_rad

    start_pos = pos.copy()
    start_heading = heading.copy()
    start_heading_angle = heading_angle

    center_direction = rotate(start_heading, turn_sign * np.pi / 2.0)
    center = start_pos + radius * center_direction
    radial_start = start_pos - center

    midpoint = center + rotate(radial_start, signed_angle * 0.5)
    end_pos = center + rotate(radial_start, signed_angle)
    end_heading = normalize(rotate(start_heading, signed_angle))
    end_heading_angle = (start_heading_angle + signed_angle) % (2.0 * np.pi)

    normal = normalize(center - midpoint)
    tangent = rotate(normal, -turn_sign * np.pi / 2.0)
    normal_angle = np.arctan2(normal[1], normal[0]) % (2.0 * np.pi)

    return {
        "number": number,
        "is_prime": is_prime,
        "turn_sign": turn_sign,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "center": center,
        "midpoint": midpoint,
        "tangent": tangent,
        "normal": normal,
        "start_heading_angle": start_heading_angle,
        "end_heading_angle": end_heading_angle,
        "end_heading": end_heading,
        "radial_start": radial_start,
        "signed_angle": signed_angle,
        "normal_angle": normal_angle,
        "resonance_angle": normal_angle,
    }
