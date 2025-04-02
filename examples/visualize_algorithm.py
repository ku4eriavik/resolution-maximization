from pathlib import Path

from algorithm import GeneticAlgorithm
from tools import create_surface_from_config, create_region_from_config,\
    create_camera_from_config, load_algorithm_config


if __name__ == '__main__':
    storage_path = Path('../data')
    conf_surf = storage_path / 'surface.json'
    conf_region = storage_path / 'region.json'
    conf_cam_list = [
        storage_path / 'camera_1.json',
        storage_path / 'camera_2.json',
        storage_path / 'camera_3.json'
    ]
    conf_alg = storage_path / 'algorithm_params.json'

    surface = create_surface_from_config(conf_surf)
    d_region = create_region_from_config(conf_region)
    cameras = [create_camera_from_config(conf_cam) for conf_cam in conf_cam_list]
    gen_conf = load_algorithm_config(conf_alg)

    solver = GeneticAlgorithm(
        surface=surface, d_region=d_region, cameras=cameras, generation_config=gen_conf
    )
    x_coordinates, res_fitness = solver.process()
    y_coordinates = surface.get_function_values(x_coordinates)
    res_coordinates = zip(x_coordinates, y_coordinates)

    print(f'Results: coordinates={res_coordinates}\tarea={res_fitness}')
