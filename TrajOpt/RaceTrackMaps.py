import numpy as np 
from matplotlib import pyplot as plt
import yaml

import LibFunctions as lib
from RaceMaps import GeneralMap


class RaceMap:
    def __init__(self, name):
        self.name = name 

        self.obs_free_hm_name = 'Maps/' + self.name + '_heatmap_obs_free.npy'
        self.obs_hm_name = 'Maps/' + self.name + '_heatmap.npy'

        self.race_course = None
        self.obs_free_hm = None
        self.obs_hm = None

        self.start = None
        self.end = None
        self.start_x = [90, 99]
        self.start_y = 50

        self.load_maps()

    def load_maps(self):
        map_name = 'Maps/' + self.name + '.npy'
        map_array = np.load(map_name)
        self.race_course = GeneralMap(map_array.T)

        self.start_x = [60, 100]
        self.start_y = 50

        self.start = [90, 51]
        self.end = [90, 45]

        # self.start_x = [0, 50]
        # self.start_y = 50

        # self.start = [20, 51]
        # self.end = [20, 45]

        obs_free_hm = self.create_hm(self.obs_free_hm_name)
        self.obs_free_hm = GeneralMap(obs_free_hm)

        self.reset_map()

    def add_obs(self, obs_locs, obs_size):
        for obs in obs_locs:
            for i in range(obs_size[0]):
                for j in range(obs_size[1]):
                    x = i + obs[0]
                    y = j + obs[1]
                    self.race_course.race_map[x, y] = 2

    def create_hm(self, hm_name, n_units=2):
        try:
            raise Exception
            return np.load(hm_name)
            # print(f"Heatmap loaded")
        except:
            hm = self._set_up_heat_map()
            np.save(hm_name, hm)
            # print(f"Heatmap saved")
            return hm

    def _set_up_heat_map(self, n=2):
        # print(f"Starting heatmap production")
        track_map = self.race_course.race_map
        for i in range(n): 
            new_map = np.zeros_like(track_map)
            # print(f"Map run: {i}")
            for i in range(1, 98 - 2):
                for j in range(1, 98 - 2):
                    left = track_map[i-1, j]
                    right = track_map[i+1, j]
                    up = track_map[i, j+1]
                    down = track_map[i, j-1]

                    # logical directions, not according to actual map orientation
                    left_up = track_map[i-1, j+1] *3
                    left_down = track_map[i-1, j-1]*3
                    right_up = track_map[i+1, j+1]*3
                    right_down = track_map[i+1, j-1]*3

                    centre = track_map[i, j]

                    obs_sum = sum((centre, left, right, up, down, left_up, left_down, right_up, right_down))
                    if obs_sum > 0:
                        new_map[i, j] = 1

            track_map = new_map
        new_map[:, 99:] = np.ones_like(new_map[:, 99:])
        new_map[:, :1] = np.ones_like(new_map[:, :1])
        new_map[:1, :] = np.ones_like(new_map[:1, :])
        new_map[99:, :] = np.ones_like(new_map[99:, :])

        # make start line
        new_map[self.start_x[0]:self.start_x[1], self.start_y] = np.ones_like(new_map[self.start_x[0]:self.start_x[1], self.start_y])
        return new_map
 
    def random_obstacles(self):
        map_name = 'Maps/' + self.name + '.npy'
        map_array = np.load(map_name)
        self.race_course = GeneralMap(map_array.T)

        obs_size = [3, 3] # representing cars 
        obs_locs = []
        for i in range(5):
            rands = lib.get_rand_ints(100-max(obs_size), 0)
            while self.obs_free_hm._check_location(rands):
                # this ensures that the cars are on the track
                rands = lib.get_rand_ints(100-max(obs_size), 0)
            obs_locs.append(rands)
        self.add_obs(obs_locs, obs_size)

    def reset_map(self):
        pass
        self.random_obstacles()

        obs_hm = self._set_up_heat_map(1)
        self.obs_hm = GeneralMap(obs_hm)

