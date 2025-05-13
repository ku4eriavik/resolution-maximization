import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Wedge, Polygon, Rectangle

from scene import SceneProcessor, PolyData
from tools import create_surface_from_config, create_region_from_config, create_camera_from_config


if __name__ == '__main__':
    conf_region_p = Path('../data/region.json')
    conf_surface_p = Path('../data/surface.json')
    cam1_p = Path('../data/camera_1.json')
    cam2_p = Path('../data/camera_2.json')
    cam3_p = Path('../data/camera_3.json')

    draw_whole = False
    draw_cropped = True

    d_region = create_region_from_config(conf_region_p)
    surface = create_surface_from_config(conf_surface_p)

    surface_points = surface.get_points()
    surface_left, surface_right = surface.get_surface_bounds()
    X = np.linspace(surface_left, surface_right, 100)
    Y = surface.get_function_values(X)

    cameras = [
        create_camera_from_config(cam1_p),
        create_camera_from_config(cam2_p),
        create_camera_from_config(cam3_p)
    ]
    # x_cameras = [3.0, 6.0, 9.0]
    # x_cameras = [4.0, 6.0, 8.0]
    # x_cameras = [2.5, 3.75, 5.0]
    # x_cameras = [2.5, 9.0, 5.5]
    # x_cameras = [5.626387, 9.151186, 2.7728794]
    # x_cameras = [2.7728794, 5.626387, 9.151186]
    # x_cameras = [2.7728794, 9.151186, 5.626387]     # opt for soft 200
    # x_cameras = [2.7783203, 9.128904, 5.6331797]    # opt for soft 100

    # used in paper
    x_cameras = [6.207, 2.974, 9.213]     # Experiment I
    # x_cameras = [9.265, 2.853, 6.408]     # Experiment II
    # x_cameras = [2.778, 9.129, 5.633]     # Experiment III
    # x_cameras = [2.773, 9.151, 5.626]     # Experiment IV

    y_cameras = surface.get_function_values(x_cameras)
    n_cameras = surface.normal_at_point(x_cameras)
    rot_cameras = [np.atan2(ny, nx) for (nx, ny) in n_cameras]
    loc_cameras = [np.asarray([x, y]) for (x, y) in zip(x_cameras, y_cameras)]
    for idx_cam, (cam, cam_rot, cam_loc) in enumerate(zip(cameras, rot_cameras, loc_cameras)):
        cam.rotate(cam_rot)
        cam.translate(cam_loc)
        print(f'Camera-{idx_cam}: rot={np.rad2deg(cam_rot)}\tloc={cam_loc}')
    print()

    scene_proc = SceneProcessor(surface=surface, cameras=cameras, d_region=d_region, approximation_count=10)
    distances = scene_proc.get_distances_between_cameras()
    print('Distances between all cameras:')
    print(distances)

    scene_intersection_points = scene_proc.trace_scene_rays()
    all_intersection_points = []
    for cam_inter_p in scene_intersection_points:
        for ray_inter_p in cam_inter_p:
            for from_inter_p, to_inter_p in ray_inter_p:
                if from_inter_p is not None:
                    all_intersection_points.append(from_inter_p)
                if to_inter_p is not None:
                    all_intersection_points.append(to_inter_p)
    all_intersection_points = np.asarray(all_intersection_points)

    scene_wedges = scene_proc.get_scene_wedges_union()
    scene_sector_angles = scene_proc.get_scene_sectors_angles()
    scene_cropped_wedges = scene_proc.crop_region()
    scene_cropped_points = scene_cropped_wedges[PolyData.POINTS]
    scene_total_area = scene_cropped_wedges[PolyData.AREA]
    print(f'Total area: {scene_total_area}')

    # visualize all
    fig = plt.figure(dpi=100, figsize=(7, 7))
    ax = fig.add_subplot()

    d_rect = Rectangle(
        (d_region[0][0], d_region[1][0]),
        width=d_region[0][1] - d_region[0][0],
        height=d_region[1][1] - d_region[1][0],
        facecolor='None', edgecolor='red', label='D region'
    )
    ax.add_patch(d_rect)

    ax.plot(X, Y, c='grey', label='Surface S(x)')
    ax.scatter(*surface_points.T, c='grey', label='Surface points')
    ax.scatter(x_cameras, y_cameras, c='blue', label='Cameras')

    for cam in cameras:
        cam_rot, cam_loc = cam.get_transform()
        cam_rays = cam.get_rays() - cam_loc
        for r in cam_rays:
            ax.plot(
                [cam_loc[0], cam_loc[0] + 100 * r[0]],
                [cam_loc[1], cam_loc[1] + 100 * r[1]],
                color='green', linewidth=1.0
            )
    ax.scatter(*all_intersection_points.T, c='black', s=5.0, label='Intersection points')

    if draw_whole:
        for idx_cam in range(len(cameras)):
            cam = cameras[idx_cam]
            cam_rot, cam_loc = cam.get_transform()

            fov_sectors_angles = scene_sector_angles[idx_cam]

            cam_wedges = scene_wedges[idx_cam]
            for idx_sec in range(len(cam_wedges)):
                angle_from, angle_to = np.rad2deg(fov_sectors_angles[idx_sec])

                sec_wedges = cam_wedges[idx_sec]
                for radius_m, radius_M in sec_wedges:
                    wedge = Wedge(
                        cam_loc, radius_M, theta1=angle_from, theta2=angle_to, width=radius_M - radius_m,
                        facecolor='blue', edgecolor='black', alpha=0.25
                    )
                    ax.add_patch(wedge)

    if draw_cropped:
        for cropped_points in scene_cropped_points:
            poly = Polygon(xy=cropped_points, facecolor='green', edgecolor='black', alpha=0.5)
            ax.add_patch(poly)

    font_size = 14
    ax.legend(fontsize=font_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X coordinate', fontsize=font_size)
    ax.set_ylabel('Y coordinate', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.set_xlim(d_region[0][0] - 0.5, d_region[0][1] + 0.5)
    ax.set_ylim(d_region[1][0] - 0.5, d_region[1][1] + 0.5)
    plt.show()
