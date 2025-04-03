import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box


class PolyData:
    POINTS = 'points'
    AREA = 'area'


class WedgesCropper:
    def __init__(self, d_region, arc_pts_count=10):
        (left, right), (bottom, top) = d_region
        self.region = box(minx=left, miny=bottom, maxx=right, maxy=top)
        self.num_pts = arc_pts_count

    def approximate_wedge(self, wedge_center, wedge_radii, wedge_rays):
        radius_min, radius_max = wedge_radii
        ray_from, ray_to = wedge_rays

        from_min = wedge_center + radius_min * ray_from
        from_max = wedge_center + radius_max * ray_from
        to_min = wedge_center + radius_min * ray_to
        to_max = wedge_center + radius_max * ray_to

        angle_from = np.arctan2(ray_from[1], ray_from[0])
        angle_to = np.arctan2(ray_to[1], ray_to[0])

        arc_angles = np.linspace(start=angle_from, stop=angle_to, num=self.num_pts)[1:-1]
        arc_vectors = np.asarray([np.cos(arc_angles), np.sin(arc_angles)]).T

        arc_min = wedge_center + radius_min * np.flip(arc_vectors, axis=0)
        arc_max = wedge_center + radius_max * arc_vectors

        return np.asarray([
            from_min, from_max, *arc_max, to_max, to_min, *arc_min
        ])

    def crop_wedge(self, wedge_center, wedge_radii, wedge_rays):
        wedge_points = self.approximate_wedge(wedge_center, wedge_radii, wedge_rays)
        wedge_poly = Polygon(wedge_points)

        cropped_poly = wedge_poly.intersection(self.region)

        if cropped_poly.is_empty:
            return None

        if isinstance(cropped_poly, MultiPolygon):
            cropped_points = []
            for poly in cropped_poly.geoms:
                if isinstance(poly, Polygon):
                    cropped_points.append(poly.exterior.coords)
            cropped_area = cropped_poly.area
        elif isinstance(cropped_poly, Polygon):
            cropped_points = np.asarray(cropped_poly.exterior.coords)
            cropped_area = cropped_poly.area
        else:
            print(cropped_poly, cropped_poly.area)
            return None

        return {
            PolyData.POINTS: cropped_points,
            PolyData.AREA: cropped_area
        }
