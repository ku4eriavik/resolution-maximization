import numpy as np
from json import load
from pathlib import Path

from surface import Surface
from camera import Camera


def create_surface_from_config(config_path: Path):
    assert config_path.exists(), f'Surface S(x) config file {str(config_path)} does not exist.'

    with open(str(config_path), 'r') as json_file:
        data = load(json_file)
        points = data.get('points', None)
        assert points is not None, 'Parameter points has to be set.'

        surface = Surface(points=np.asarray(points, dtype=np.float32))
        return surface


def create_region_from_config(config_path: Path):
    assert config_path.exists(), f'Region D config file {str(config_path)} does not exist.'

    with open(str(config_path), 'r') as json_file:
        data = load(json_file)
        x_range = data.get('x_range', None)
        y_range = data.get('y_range', None)
        assert x_range is not None, 'Parameter x_range has to be set.'
        assert y_range is not None, 'Parameter y_range has to be set.'

        return x_range, y_range


def create_camera_from_config(config_path: Path):
    assert config_path.exists(), f'Camera C config file {str(config_path)} does not exist.'

    with open(str(config_path), 'r') as json_file:
        data = load(json_file)
        fov_deg = data.get('fov_deg', None)
        res_pix = data.get('res_pix', None)
        assert fov_deg is not None, 'Parameter fov_deg has to be set.'
        assert res_pix is not None, 'Parameter res_pix has to be set.'

        camera = Camera(fov_rad=np.deg2rad(fov_deg), res_pix=res_pix)
        return camera


def load_algorithm_config(config_path: Path):
    assert config_path.exists(), f'Algorithm config file {str(config_path)} does not exist.'

    with open(str(config_path), 'r') as json_file:
        config = load(json_file)
        return config
