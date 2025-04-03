import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from matplotlib.patches import Polygon as mpl_poly

from wedges import WedgesCropper, PolyData


if __name__ == '__main__':
    D = [2.0, 6.0], [4.0, 10.0]
    sector_center = np.asarray([3.0, 4.5])
    ray_from = np.asarray([2.0, 1.0]) / np.sqrt(5.0)
    ray_to = np.asarray([1.0, 2.0]) / np.sqrt(5.0)
    angle_from = np.atan2(ray_from[1], ray_from[0])
    angle_to = np.atan2(ray_to[1], ray_to[0])
    seg_un = [
        (2.0, 5.0), (6.0, 7.0), (8.0, 12.0), (13.0, 20.0)
    ]
    w_cropper = WedgesCropper(d_region=D, arc_pts_count=10)
    wedges_points = []
    cropped_wedges_polygons = []
    for (rad_m, rad_M) in seg_un:
        w_points = w_cropper.approximate_wedge(
            wedge_center=sector_center, wedge_radii=(rad_m, rad_M), wedge_rays=(ray_from, ray_to)
        )
        wedges_points.append(w_points)

        crop_w_poly = w_cropper.crop_wedge(
            wedge_center=sector_center, wedge_radii=(rad_m, rad_M), wedge_rays=(ray_from, ray_to)
        )
        cropped_wedges_polygons.append(crop_w_poly)

    # visualize all
    fig = plt.figure(dpi=100, figsize=(7, 7))
    ax = fig.add_subplot()

    d_rect = Rectangle(
        [D[0][0], D[1][0]], width=D[0][1] - D[0][0], height=D[1][1] - D[1][0],
        facecolor='None', edgecolor='red', label='D region'
    )
    ax.add_patch(d_rect)

    ax.scatter([sector_center[0]], [sector_center[1]], c='grey', label='Camera')

    line_from = ax.axline(sector_center, slope=np.tan(angle_from), linestyle='--', c='grey', label='Ray from')
    line_from.set_clip_path(d_rect)

    line_to = ax.axline(sector_center, slope=np.tan(angle_to), linestyle='--', c='grey', label='Ray to')
    line_to.set_clip_path(d_rect)

    for idx_w, (rad_m, rad_M) in enumerate(seg_un):
        w_points = wedges_points[idx_w]
        crop_w_poly = cropped_wedges_polygons[idx_w]

        wedge = Wedge(
            sector_center, rad_M, np.rad2deg(angle_from), np.rad2deg(angle_to), width=rad_M - rad_m,
            facecolor='green', edgecolor='black', alpha=0.5
        )
        ax.add_patch(wedge)

        ax.scatter(*w_points.T, c='black', s=5.0)

        if crop_w_poly is not None:
            crop_w_points = crop_w_poly[PolyData.POINTS]

            poly = mpl_poly(xy=crop_w_points, facecolor='blue', edgecolor='black', alpha=0.5)
            ax.add_patch(poly)

            ax.scatter(*crop_w_points.T, c='blue', s=3.0)

    ax.legend()
    ax.set_aspect('equal')
    plt.show()
