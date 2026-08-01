import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

from .constants import VALID_PRIME_DISPLAY_MODES


def center_layer_color(center):
    layer = int(round(center[0])) + int(round(center[1]))
    return {1: "#00aa00", 3: "#ff0000", 5: "#0066ff"}.get(layer, "#000000")


def draw_prime_points(ax, data, prime_display_mode, prime_point_size, max_labels):
    if not data["prime_plot_points"]:
        return
    if prime_display_mode not in VALID_PRIME_DISPLAY_MODES:
        raise ValueError(
            "prime_display_mode muss 'points', 'labels', 'points+labels' oder 'none' sein."
        )
    if prime_display_mode == "none":
        return

    points = np.asarray(data["prime_plot_points"], dtype=np.float64)

    if "points" in prime_display_mode:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=prime_point_size,
            c=data["prime_plot_colors"],
            edgecolors="none",
            zorder=10,
        )

    if "labels" in prime_display_mode:
        for i in range(min(len(points), max_labels)):
            ax.text(
                points[i, 0],
                points[i, 1],
                data["prime_plot_labels"][i],
                fontsize=5,
                color=data["prime_plot_colors"][i],
                zorder=12,
            )


def draw_family_overlays(
    ax,
    data,
    family_centroids,
    show_family_lines,
    show_family_centers,
    family_center_size,
    show_family_centroids,
    show_between_family_lines,
):
    if show_family_lines:
        for points in data["family_line_points"].values():
            if len(points) < 2:
                continue
            arr = np.asarray(points, dtype=np.float64)
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                linewidth=0.5,
                alpha=0.65,
                color="#444444",
                zorder=8,
            )

    if show_family_centers and data["family_centers_raw"]:
        centers = np.asarray(
            list(data["family_centers_raw"].values()), dtype=np.float64
        )
        sizes = np.asarray(
            [
                (
                    1.5 * family_center_size
                    if len(data["prime_families"][key]) > 1
                    else family_center_size
                )
                for key in data["family_centers_raw"]
            ],
            dtype=np.float64,
        )
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=sizes,
            marker="o",
            facecolors="none",
            edgecolors="#ffd500",
            linewidths=0.7,
            zorder=9,
        )

    if show_family_centroids and family_centroids:
        centers = np.asarray(list(family_centroids.values()), dtype=np.float64)
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=family_center_size,
            marker="x",
            color="#7a00ff",
            linewidths=0.7,
            zorder=9,
        )

    if show_between_family_lines and len(family_centroids) > 1:
        centers = np.asarray(list(family_centroids.values()), dtype=np.float64)
        ax.plot(
            centers[:, 0],
            centers[:, 1],
            linewidth=0.35,
            alpha=0.35,
            color="#222222",
            zorder=5,
        )


def draw_reference_points(
    ax,
    centroid,
    pos,
    show_start_point,
    show_end_point,
    show_centroid,
    centroid_size,
):
    if show_start_point:
        ax.scatter(
            [0.0],
            [0.0],
            s=70,
            marker="o",
            color="#ff0040",
            edgecolors="black",
            linewidths=0.5,
            zorder=20,
        )

    if show_end_point:
        ax.scatter(
            [pos[0]],
            [pos[1]],
            s=70,
            marker="s",
            color="#00aaff",
            edgecolors="black",
            linewidths=0.5,
            zorder=20,
        )

    if show_centroid:
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=centroid_size,
            marker="X",
            color="yellow",
            edgecolors="black",
            linewidths=0.7,
            zorder=21,
        )


def draw_plot(
    data,
    centroid,
    family_centroids,
    pos,
    N,
    largest_family,
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
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if show_arcs:
        for color, arc_data in data["arc_groups"].items():
            ax.add_collection(
                LineCollection(
                    arc_data["segments"],
                    colors=color,
                    linewidths=arc_data["linewidth"],
                    capstyle="round",
                    joinstyle="round",
                    zorder=1,
                )
            )

    if show_local_normals and data["normal_segments"]:
        ax.add_collection(
            LineCollection(
                data["normal_segments"],
                colors="#666666",
                linewidths=0.35,
                alpha=0.5,
                zorder=3,
            )
        )

    if show_circle_centers and data["circle_centers"]:
        centers = np.asarray(data["circle_centers"], dtype=np.float64)
        colors = [center_layer_color(center) for center in data["circle_centers"]]
        ax.scatter(centers[:, 0], centers[:, 1], s=8, alpha=0.8, c=colors, zorder=4)

    if show_arc_midpoints and data["arc_midpoints"]:
        midpoints = np.asarray(data["arc_midpoints"], dtype=np.float64)
        ax.scatter(
            midpoints[:, 0],
            midpoints[:, 1],
            s=3,
            alpha=0.35,
            color="#999999",
            zorder=4,
        )

    draw_prime_points(ax, data, prime_display_mode, prime_point_size, max_labels)
    draw_family_overlays(
        ax,
        data,
        family_centroids,
        show_family_lines,
        show_family_centers,
        family_center_size,
        show_family_centroids,
        show_between_family_lines,
    )
    draw_reference_points(
        ax,
        centroid,
        pos,
        show_start_point,
        show_end_point,
        show_centroid,
        centroid_size,
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.margins(0.03)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    # ax.set_title(
    #     f"Rekursive Primzahl-Geometrie | N={N} | "
    #     f"Familien={len(data['prime_families'])} | groesste Familie={largest_family}"
    # )
    ax.set_title(f"Rekursive Primzahl-Geometrie\n" f"N = {N:,}")
    # ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.yaxis.set_major_locator(MultipleLocator(5))

    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    # ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    # ax.grid(True, which="minor", alpha=0.08, linewidth=0.4)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    plt.show()
    return fig, ax
