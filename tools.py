import numpy as np
from json import load
from pathlib import Path

from surface import Surface


def create_surface_from_config(config_path: Path):
    assert config_path.exists(), f'Surface S(x) config file {str(config_path)} does not exist.'

    with open(str(config_path), 'r') as json_file:
        data = load(json_file)
        points = data.get('points', None)
        assert points is not None, 'Parameter points has to be set.'

        surface = Surface(points=np.asarray(points, dtype=np.float32))
        return surface
