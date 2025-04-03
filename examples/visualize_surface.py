import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tools import create_surface_from_config


if __name__ == '__main__':
    conf_p = Path('../data/surface.json')

    surf = create_surface_from_config(config_path=conf_p)
    surf_points = surf.get_points()
    surf_left, surf_right = surf.get_surface_bounds()

    X = np.linspace(surf_left, surf_right, 100)
    Y = surf.get_function_values(X)

    x_pivot = np.random.choice(X)
    y_pivot = surf.get_function_values(x_pivot)
    t_pivot = surf.tangent_at_point(x_pivot)
    n_pivot = surf.normal_at_point(x_pivot)

    dist_1_2 = surf.arc_length(surf_points[1][0], surf_points[2][0])
    dist_1_3 = surf.arc_length(surf_points[1][0], surf_points[3][0])
    print(dist_1_2, dist_1_3)

    fig = plt.figure(dpi=100, figsize=(7, 7))
    ax = fig.add_subplot()

    ax.scatter(*surf_points.T, c='grey', label='Surface points')
    ax.plot(X, Y, c='grey', label='Surface')

    ax.plot([x_pivot, x_pivot + t_pivot[0]], [y_pivot, y_pivot + t_pivot[1]], c='red', label='Tangent')
    ax.plot([x_pivot, x_pivot + n_pivot[0]], [y_pivot, y_pivot + n_pivot[1]], c='red', label='Normal')

    ax.legend()
    ax.set_aspect('equal')
    plt.show()
