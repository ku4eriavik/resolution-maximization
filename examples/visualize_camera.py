import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tools import create_camera_from_config


if __name__ == '__main__':
    conf_p = Path('../data/camera_2.json')

    cam_rot = np.deg2rad(30.0)
    cam_loc = np.asarray([-1.5, 2.0])

    cam = create_camera_from_config(conf_p)
    cam.rotate(cam_rot)
    cam.translate(cam_loc)

    cam_rays = cam.get_rays()

    fig = plt.figure(dpi=100, figsize=(7, 7))
    ax = fig.add_subplot()

    ax.plot([cam_loc[0], cam_rays[0][0]], [cam_loc[1], cam_rays[0][1]], color='green')
    for r in cam_rays[1:-1]:
        ax.plot([cam_loc[0], r[0]], [cam_loc[1], r[1]], color='green', linestyle='--')
    ax.plot([cam_loc[0], cam_rays[-1][0]], [cam_loc[1], cam_rays[-1][1]], color='green')

    ax.set_aspect('equal')
    plt.show()
