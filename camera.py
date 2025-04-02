import numpy as np


class Camera:
    def __init__(self, fov_rad, res_pix):
        self._rotation = 0.0
        self._location = 0.0, 0.0

        self._fov = fov_rad
        self._angle_res = self._fov / res_pix
        self._rays = []

        for i in range(1 + res_pix):
            ray_vector = np.asarray([1.0, 0.0])
            ray_angle = i * self._angle_res - self._fov / 2.0
            rot_matrix = np.asarray([
                [np.cos(ray_angle), -np.sin(ray_angle)],
                [np.sin(ray_angle), np.cos(ray_angle)]
            ])
            ray_vector = rot_matrix @ ray_vector
            self._rays.append(ray_vector)

    def rotate(self, angle_rad):
        self._rotation = angle_rad

    def translate(self, shift):
        self._location = shift

    def get_transform(self):
        return self._rotation, self._location

    def get_rays(self):
        res_rays = []
        for ray in self._rays:
            matrix = np.array([
                [np.cos(self._rotation), -np.sin(self._rotation)],
                [np.sin(self._rotation), np.cos(self._rotation)]
            ])
            res_ray = matrix @ ray + self._location
            res_rays.append(res_ray)
        return np.asarray(res_rays)
