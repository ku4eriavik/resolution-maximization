import random
import numpy as np
from time import time
from deap import base, creator, tools

from surface import Surface
from camera import Camera
from scene import SceneProcessor, PolyData


class GeneticAlgorithm:
    def __init__(self, surface: Surface, d_region: tuple, cameras: list[Camera], generation_config: dict):
        self._surface = surface

        self._d_region = d_region
        self._cameras = cameras

        self._gen_current = 0
        self._gen_count = generation_config['amount_generations']
        self._min_distance = generation_config['minimal_distance']
        self._normal_sigma = self._min_distance / 2.0
        self._use_soft_penalty = generation_config['use_soft_penalty']
        self._w_penalty = generation_config['penalty_weight']
        self._approx_count = generation_config['approximation_count']
        self._size_pop = generation_config['size_population']
        self._size_elite = generation_config['size_elite']
        self._size_plebs = generation_config['size_plebs']
        self._error_rate = generation_config['error_rate']

        loops = np.identity(len(self._cameras), dtype=np.float32)
        self._loops = loops * (self._min_distance + self._error_rate)

        self.__register_methods()

    def __register_methods(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        self._toolbox = base.Toolbox()
        self._toolbox.register(alias='individual', function=self.generate_initial_individual)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)

        self._toolbox.register(alias='mutate', function=self.mutate_dynamic)
        self._toolbox.register(alias='mate', function=self.crossover_dynamic)
        self._toolbox.register(alias='evaluate', function=self.evaluate)
        self._toolbox.register(alias='select', function=self.select)

        self._population = None

    @staticmethod
    def __to_apply(probability):
        return random.random() < probability

    def __calculate_penalty(self, distances):
        penalty = 0.0
        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):
                if distances[i, j] < self._min_distance:
                    penalty += self._min_distance - distances[i, j]
        return penalty

    def __has_penalty(self, distances):
        return np.any(self._loops + distances < self._min_distance)

    def generate_initial_individual(self):
        left, right = self._surface.get_surface_bounds()
        amount = len(self._cameras)
        coordinates = [random.uniform(left, right) for _ in range(amount)]
        random.shuffle(coordinates)
        return creator.Individual(coordinates)

    def mutate_dynamic(self, individual):
        progress = self._gen_current / self._gen_count

        apply_uniform = self.__to_apply(1.0 - progress)
        apply_swap = self.__to_apply(0.5)
        apply_normal = self.__to_apply(progress)

        left, right = self._surface.get_surface_bounds()
        amount = len(self._cameras)

        if apply_uniform:
            for i in range(amount):
                if self.__to_apply(0.5):
                    individual[i] = random.uniform(left, right)

        if apply_swap:
            i, j = random.sample(range(amount), 2)
            individual[i], individual[j] = individual[j], individual[i]

        if apply_normal:
            for i in range(amount):
                if self.__to_apply(0.5):
                    shift = random.gauss(0.0, self._normal_sigma)
                    individual[i] = max(left, min(right, individual[i] + shift))
        return individual,

    def crossover_dynamic(self, individual1, individual2):
        amount = len(self._cameras)
        crossover = random.choice(['apply_one_point', 'apply_uniform'])

        if crossover == 'apply_one_point':
            one_point = random.randint(1, amount - 1)
            individual1[one_point:], individual2[one_point:] = individual2[one_point:], individual1[one_point:]

        if crossover == 'apply_uniform':
            for i in range(amount):
                if self.__to_apply(0.5):
                    individual1[i], individual2[i] = individual2[i], individual1[i]
        return individual1, individual2

    def evaluate(self, individual):
        x_cameras = individual[:]
        y_cameras = self._surface.get_function_values(x_cameras)
        n_cameras = self._surface.normal_at_point(x_cameras)

        for cam, x, y, n in zip(self._cameras, x_cameras, y_cameras, n_cameras):
            cam_loc = np.asarray([x, y])
            cam_rot = np.atan2(n[1], n[0])
            cam.rotate(cam_rot)
            cam.translate(cam_loc)

        scene_proc = SceneProcessor(
            surface=self._surface, cameras=self._cameras,
            d_region=self._d_region, approximation_count=self._approx_count
        )

        distances = scene_proc.get_distances_between_cameras()
        if self._use_soft_penalty:
            penalty = self.__calculate_penalty(distances)
        else:
            penalty = float('inf') if self.__has_penalty(distances) else 0.0

        scene_proc.trace_scene_rays()
        scene_cropped_wedges = scene_proc.crop_region()
        scene_total_area = scene_cropped_wedges[PolyData.AREA]

        score = scene_total_area - self._w_penalty * penalty
        return score,

    def select(self, population):
        elite = tools.selBest(population, k=self._size_elite)
        rest = [ind for ind in population if ind not in elite]
        plebs = tools.selTournament(rest, k=self._size_plebs, tournsize=3)
        return elite + plebs

    def process(self):
        self._population = self._toolbox.population(n=self._size_pop)

        best_fit = float('-inf')
        best_ind = None

        for self._gen_current in range(1, self._gen_count + 1):
            print(f'Generation\t{self._gen_current}/{self._gen_count}')

            t_before = time()

            offspring = self._toolbox.select(self._population)
            offspring = list(map(self._toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if self.__to_apply(0.5):
                    self._toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if self.__to_apply(0.5):
                    self._toolbox.mutate(mutant)
                    del mutant.fitness.values

            for candidate in offspring:
                if not candidate.fitness.valid:
                    candidate.fitness.values = self._toolbox.evaluate(candidate)

            self._population[:] = offspring

            fits = [ind.fitness.values[0] for ind in self._population]
            max_fit = max(fits)
            avg_fit = sum(fits) / len(fits)

            if max_fit > best_fit:
                best_fit = max_fit
                best_ind = self._population[np.argmax(fits)]

            t_after = time()
            t_elapsed = t_after - t_before

            print(f'\tmax-fit={max_fit}\t\tavg-fit={avg_fit}')
            print(f'\tbest-fit={best_fit}\t\tbest-ind={best_ind}')
            print(f'\ttime-elapsed={t_elapsed} sec')
            print()

        return best_ind, best_fit
