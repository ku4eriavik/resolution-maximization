import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline


class Surface:
    def __init__(self, points):
        self._points = points

        x, y = self._points.T
        self._spline = CubicSpline(x, y, bc_type='natural')
        self._derivative = self._spline.derivative()

    def get_points(self):
        return self._points

    def get_surface_bounds(self):
        x, _ = self._points.T
        return x[0], x[-1]

    def get_function_values(self, x_values):
        return self._spline(x_values)

    def get_1_derivative_values(self, x_values):
        return self._derivative(x_values)

    def tangent_at_point(self, x_point):
        dy_dx = self._derivative(x_point)
        magnitude = (1.0 + dy_dx ** 2) ** 0.5
        tangent = np.asarray([1.0 / magnitude, dy_dx / magnitude]).T
        return tangent

    def normal_at_point(self, x_point):
        dy_dx = self._derivative(x_point)
        magnitude = (1.0 + dy_dx ** 2) ** 0.5
        normal = np.asarray([-dy_dx / magnitude, 1.0 / magnitude]).T
        return normal

    def __arc_length(self, x):
        return np.sqrt(1.0 + self._derivative(x) ** 2)

    def arc_length(self, x1, x2):
        length, _ = quad(self.__arc_length, x1, x2)
        return np.abs(length)
