import numpy as np 
from matplotlib import pyplot as plt
import yaml

import LibFunctions as lib

class GeneralMap:
    def __init__(self, race_map):
        self.race_map = race_map

        self.map_height = race_map.shape[0]
        self.map_width = race_map.shape[1]

    def _check_location(self, x):
        if x[0] < 1 or x[0] > self.map_width  - 1:
            return True
        if x[1] < 1 or x[1] > self.map_height - 1:
            return True 

        if self.race_map[int(round(x[0])), int(round(x[1]))]:
            return True

        return False

    def _check_line(self, start, end):
        n_checks = 5
        dif = lib.sub_locations(end, start)
        diff = [dif[0] / (n_checks), dif[1] / n_checks]
        for i in range(5):
            search_val = lib.add_locations(start, diff, i + 1)
            if self._check_location(search_val):
                return True
        return False

    def show_map(self, show=False, path=None):
        fig = plt.figure(7)
        plt.clf()

        plt.imshow(self.race_map.T, origin='lower')
        # plt.plot(self.start[0], self.start[1], '*', markersize=20)
        # plt.plot(self.end[0], self.end[1], '*', markersize=20)

        if path is not None:
            xs, ys = [], []
            for pt in path:
                xs.append(pt[0])
                ys.append(pt[1])
            
            plt.plot(xs, ys)
            plt.plot(xs, ys, 'x', markersize=16)

        plt.pause(0.001)
        if show:
            plt.show()


class EnvironmentMap:
    def __init__(self, name):
        self.name = name 
        self.yaml_name = 'Maps/' + self.name + '.yaml'

        self.obs_free_hm_name = 'Maps/' + self.name + '_heatmap_obs_free.npy'
        self.hm_name = 'Maps/' + self.name + '_heatmap.npy'

        self.race_course = None
        self.obs_hm = None
        self.obs_free_hm = None

        self.start = None
        self.end = None

        self.load_maps()

    def load_maps(self):
        yaml_editor = MapYamlEditor(self.name)
        yaml_editor.load_yaml_file()
        # yaml_editor.print_contents()

        self.start = yaml_editor.start
        self.end = yaml_editor.end

        map_name = 'Maps/' + yaml_editor.map_name + '.npy'
        map_array = np.load(map_name)
        self.race_course = GeneralMap(map_array.T)

        obs_free_hm = self.create_hm(self.obs_free_hm_name)
        self.obs_free_hm = GeneralMap(obs_free_hm)

        self.add_obs(yaml_editor.obs_locs, yaml_editor.obs_size)
        obs_hm = self.create_hm(self.hm_name)
        self.obs_hm = GeneralMap(obs_hm)

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

    def _set_up_heat_map(self):
        # print(f"Starting heatmap production")
        track_map = self.race_course.race_map
        for i in range(2): 
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
        new_map[:, 98:] = np.ones_like(new_map[:, 98:])
        return new_map
 
    def generate_random_start(self):
        self.start = lib.get_rands(90, 5)
        while self.obs_hm._check_location(self.start):
            self.start = lib.get_rands(90, 5)

        self.end = lib.get_rands()
        while self.obs_hm._check_location(self.end) or \
            lib.get_distance(self.start, self.end) < 30:
            self.end = lib.get_rands(90, 5)
            # self.end = lib.get_rands(80, 10)

    def random_obstacles(self):
        self.race_course.race_map = np.zeros((100, 100))
        obs_size = [6, 8]
        obs_locs = []
        for i in range(5):
            obs_locs.append(lib.get_rand_ints(40, 25))
        self.add_obs(obs_locs, obs_size)

    def set_start(self):
        self.start = [50, 15]
        self.end = [50, 85]

    def reset_map(self):
        self.generate_random_start()


class MapYamlEditor:
    def __init__(self, map_name='TestTrack1000'):
        self.dict = {}

        self.map_name = map_name
        self.start = None
        self.end = None
        self.obs_locs = []
        self.obs_size = None
        self.height = None
        self.width = None

        self.set_dict_values()
        self.create_dictionary()

    def set_dict_values(self):
        self.start = [53, 5]
        self.end = [90, 20]

        self.obs_locs = [[40, 75], [25, 45], [70, 60], [50, 30], [75, 15], [10, 80], [20, 10]]

        self.obs_size = [8, 14]
        self.height = 100
        self.width = 100

    def create_dictionary(self):
        self.dict['map_name'] = self.map_name
        self.dict['start'] = self.start
        self.dict['end'] = self.end

        obs_dict = {}
        for i in range(len(self.obs_locs)):
            obs_dict[i] = self.obs_locs[i]

        self.dict['obstacles'] = obs_dict
        self.dict['obs_size'] = self.obs_size
        self.dict['width'] = self.width
        self.dict['height'] = self.height

    def save_dict_yaml(self):
        yaml_name = 'Maps/' + self.map_name + '.yaml'
        with open(yaml_name, 'w') as yaml_file:
            yaml.dump(self.dict, yaml_file)

    def load_yaml_file(self):
        yaml_name = 'Maps/' + self.map_name + '.yaml'
        with open(yaml_name) as yaml_file:
            self.dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        self.convert_dict_values()

    def convert_dict_values(self):
        self.map_name = self.dict['map_name']
        self.start = self.dict['start']
        self.end = self.dict['end']

        obs_dict = self.dict['obstacles']
        self.obs_locs = []
        for i in range(len(obs_dict)):
            self.obs_locs.append(obs_dict[i])

        self.obs_size =  self.dict['obs_size']
        self.width = self.dict['width']
        self.height = self.dict['height']

    def print_contents(self):
        print(f"Dictionary of: {self.map_name}")
        print(self.dict)




#Test and setup fcns
def create_yaml_file():
    yaml_maker = MapYamlEditor('TrainTrackEmpty')
    yaml_maker.save_dict_yaml()
    yaml_maker.load_yaml_file()
    yaml_maker.print_contents()

    my_environ = EnvironmentMap('TrainTrackEmpty')
    my_environ.load_maps()
    my_environ.race_course.show_map()

def test_environment():
    my_environ = EnvironmentMap('TestTrack1000')
    my_environ.load_maps()
    my_environ.race_course.show_map()
    # my_environ.obs_free_hm.show_map()
    # my_environ.obs_hm.show_map()
    # my_environ.race_course.show_map()


if __name__ == "__main__":
    create_yaml_file()    
    # test_environment()

    


