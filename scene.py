import numpy as np

from surface import Surface
from camera import Camera
from segment_union import SegmentsUnion
from wedges import WedgesCropper, PolyData


class SectorProcessor:
    def __init__(self, location, ray_from, ray_to):
        self._location = location
        self._ray_from = ray_from
        self._ray_to = ray_to
        self._seg_un = SegmentsUnion()

    def get_sector_angles(self):
        x_from, y_from = self._ray_from
        x_to, y_to = self._ray_to

        angle_from = np.atan2(y_from, x_from)
        angle_to = np.atan2(y_to, x_to)
        return angle_from, angle_to

    @staticmethod
    def _intersect_ray(center_1, direction_1, center_2, direction_2):
        A = np.column_stack([-direction_1, direction_2])
        b = center_1 - center_2

        if np.linalg.matrix_rank(A) < 2:
            if np.linalg.matrix_rank(np.column_stack([A, b])) < 2:
                return None     # coincide
            else:
                return None     # parallel

        t, u = np.linalg.solve(A, b)
        if t >= 0.0 and u >= 0.0:
            return center_1 + t * direction_1   # intersect
        return None                             # intersect lines, not rays

    def trace_next_ray(self, ray_center, ray_direction):
        inter_from = self._intersect_ray(self._location, self._ray_from, ray_center, ray_direction)
        inter_to = self._intersect_ray(self._location, self._ray_to, ray_center, ray_direction)

        if inter_from is not None and inter_to is not None:
            radius_from = np.linalg.norm(self._location - inter_from)
            radius_to = np.linalg.norm(self._location - inter_to)
            self._seg_un.add([min(radius_from, radius_to), max(radius_from, radius_to)])

        return inter_from, inter_to

    def get_sector_wedges_union(self):
        return self._seg_un.get_union()


class FovProcessor:
    def __init__(self, location, fov_rays):
        self._sectors_processors = []
        self._fov_sectors_angles = []

        for idx in range(1, len(fov_rays)):
            ray_from = fov_rays[idx - 1]
            ray_to = fov_rays[idx]
            sec_proc = SectorProcessor(location, ray_from, ray_to)
            sec_angles = sec_proc.get_sector_angles()

            self._sectors_processors.append(sec_proc)
            self._fov_sectors_angles.append(sec_angles)

    def get_fov_sectors_angles(self):
        return self._fov_sectors_angles

    def trace_next_ray(self, ray_center, ray_direction):
        intersections = []
        for sec_proc in self._sectors_processors:
            inter_from, inter_to = sec_proc.trace_next_ray(ray_center, ray_direction)
            intersections.append([inter_from, inter_to])
        return intersections

    def get_fov_wedges_union(self):
        fov_union = []
        for sec_proc in self._sectors_processors:
            sec_union = sec_proc.get_sector_wedges_union()
            fov_union.append(sec_union)
        return fov_union


class SceneProcessor:
    def __init__(self, surface: Surface, cameras: list[Camera], d_region, approximation_count: int):
        self._surface = surface
        self._cameras = cameras

        self._wedges_cropper = WedgesCropper(d_region, arc_pts_count=approximation_count)

        self._fov_processors = []
        self._scene_sectors_angles = []

        for cam in self._cameras:
            cam_rot, cam_loc = cam.get_transform()
            cam_rays = cam.get_rays() - cam_loc
            fov_proc = FovProcessor(cam_loc, cam_rays)
            fov_sec_angles = fov_proc.get_fov_sectors_angles()

            self._fov_processors.append(fov_proc)
            self._scene_sectors_angles.append(fov_sec_angles)

    def get_scene_sectors_angles(self):
        return self._scene_sectors_angles

    def trace_scene_rays(self):
        scene_intersections = []
        for i in range(len(self._cameras)):
            cam_intersections = []

            for j in range(len(self._cameras)):
                if i == j:
                    continue

                fov_proc = self._fov_processors[i]
                cam = self._cameras[j]

                cam_rot, cam_loc = cam.get_transform()
                cam_rays = cam.get_rays()

                for ray in cam_rays:
                    unit_ray = ray - cam_loc
                    ray_intersections = fov_proc.trace_next_ray(cam_loc, unit_ray)
                    cam_intersections.append(ray_intersections)

            scene_intersections.append(cam_intersections)
        return scene_intersections

    def get_scene_wedges_union(self):
        scene_wedges_union = []
        for fov_proc in self._fov_processors:
            fov_wedges_union = fov_proc.get_fov_wedges_union()
            scene_wedges_union.append(fov_wedges_union)
        return scene_wedges_union

    def crop_region(self):
        scene_wedges_union = self.get_scene_wedges_union()

        scene_area = 0.0
        scene_cropped_poly_points = []
        for idx_cam, fov_wedges_union in enumerate(scene_wedges_union):
            cam = self._cameras[idx_cam]
            _, cam_loc = cam.get_transform()
            cam_rays = cam.get_rays() - cam_loc

            for idx_sec, sector_wedges_union in enumerate(fov_wedges_union):
                sec_rays = cam_rays[idx_sec], cam_rays[idx_sec + 1]

                for wedge_radii in sector_wedges_union:
                    wedge_poly = self._wedges_cropper.crop_wedge(
                        wedge_center=cam_loc, wedge_radii=wedge_radii, wedge_rays=sec_rays
                    )

                    if wedge_poly is None:
                        break

                    wedge_points = wedge_poly[PolyData.POINTS]
                    if isinstance(wedge_points, np.ndarray):    # Polygon
                        scene_cropped_poly_points.append(wedge_points)
                    else:                                       # MultiPolygon
                        for points in wedge_points:
                            scene_cropped_poly_points.append(points)
                    scene_area += wedge_poly[PolyData.AREA]

        return {
            PolyData.POINTS: scene_cropped_poly_points,
            PolyData.AREA: scene_area
        }

    def get_distances_between_cameras(self):
        distance_matrix = np.zeros(shape=(len(self._cameras), len(self._cameras)), dtype=np.float32)
        for i in range(len(self._cameras)):
            for j in range(i + 1, len(self._cameras)):
                _, i_loc = self._cameras[i].get_transform()
                _, j_loc = self._cameras[j].get_transform()
                distance_matrix[i, j] = distance_matrix[j, i] = self._surface.arc_length(i_loc[0], j_loc[0])
        return distance_matrix
