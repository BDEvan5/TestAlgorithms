import numpy as np
from matplotlib import pyplot as plt 



class TrackMap:
    def __init__(self, csv_map="TrackMap1000"):
        self.name = csv_map

        self.track = None
        self.track_pts = None
        self.nvecs = None
        self.ws = None
        self.N = None

        self.start = None
        self.end = None

        self.obs_map = None
        self.scan_map = None
        self.obs_res = 0.1

        self.load_map()
        self.set_up_scan_map()
        lengths = [lib.get_distance(self.track_pts[i], self.track_pts[i+1]) for i in range(self.N-1)]
        lengths.insert(0, 0)
        self.cum_lengs = np.cumsum(lengths)

        self.wpts = None # used for the target
        self.target = None

    def load_map(self):
        track = []
        filename = 'Maps/' + self.name + ".csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded")

        self.track = track
        self.N = len(track)
        self.track_pts = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.ws = track[:, 4:6]

        self.start = self.track_pts[0] - 0.1
        self.end = self.track_pts[-1]

        self.random_obs(0)

    def get_min_curve_path(self):
        path_name = 'Maps/' + self.name + "_path.npy"
        try:
            # raise Exception
            path = np.load(path_name)
            print(f"Path loaded from file: min curve")
        except:
            track = self.track
            n_set = MinCurvatureTrajectory(track, self.obs_map)
            deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
            path = track[:, 0:2] + deviation

            np.save(path_name, path)
            print(f"Path saved: min curve")

        return path

    def find_nearest_point(self, x):
        distances = [lib.get_distance(x, self.track_pts[i]) for i in range(self.N)]

        nearest_idx = np.argmin(np.array(distances))

        return nearest_idx

    def _check_location(self, x):
        idx = self.find_nearest_point(x)
        dis = lib.get_distance(self.track_pts[idx], x)
        if dis > self.ws[idx, 0] * 1.5:
            return True
        return False

    def random_obs(self, n=10):
        resolution = 100
        self.obs_map = np.zeros((resolution, resolution))
        obs_size = [3, 4]
        rands = np.random.randint(1, self.N-1, n)
        obs_locs = []
        for i in range(n):
            # obs_locs.append(lib.get_rand_ints(40, 25))
            pt = self.track_pts[rands[i]][:, None]
            obs_locs.append(pt[:, 0])

        for obs in obs_locs:
            for i in range(0, obs_size[0]):
                for j in range(0, obs_size[1]):
                    x = min(int(round(i + obs[0]/ self.obs_res)), 99)
                    y = min(int(round(j + obs[1]/ self.obs_res)), 99)
                    self.obs_map[x, y] = 1

    def set_up_scan_map(self):
        try:
            # raise Exception
            self.scan_map = np.load("Maps/scan_map.npy")
        except:
            resolution = 100
            self.scan_map = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    ii = i*self.obs_res
                    jj = j*self.obs_res
                    if self._check_location([ii, jj]):
                        self.scan_map[i, j] = 1
            np.save("Maps/scan_map", self.scan_map)

            print("Scan map ready")
        # plt.imshow(self.scan_map.T)
        # plt.show()

    def get_show_map(self):
        ret_map  = np.clip(self.obs_map + self.scan_map, 0 , 1)
        return ret_map

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True

        y = int(max(min(x_in[1] / self.obs_res, 99), 0))
        x = int(max(min(x_in[0] / self.obs_res, 99), 0))
        if self.scan_map[x, y]:
            return True
        if self.obs_map[x, y]:
            return True
        return False

    def reset_map(self):
        self.random_obs(10)

    def get_s_progress(self, x):
        idx = self.find_nearest_point(x)

        if idx == 0:
            return lib.get_distance(x, self.track_pts[0])

        if idx == self.N-1:
            s = self.cum_lengs[-2] + lib.get_distance(x, self.track_pts[-2])
            return s

        p_d = lib.get_distance(x, self.track_pts[idx-1])
        n_d = lib.get_distance(x, self.track_pts[idx+1])

        if p_d < n_d:
            s = self.cum_lengs[idx-1] + p_d
        else:
            s = self.cum_lengs[idx] + lib.get_distance(self.track_pts[idx], x)


        return s

    def set_wpts(self, wpts):
        self.wpts = wpts

    def find_target(self, obs):
        distances = [lib.get_distance(obs[0:2], self.wpts[i]) for i in range(len(self.wpts))]
        ind = np.argmin(distances)
        N = len(self.wpts)

        look_ahead = 3
        pind = ind + look_ahead
        if pind >= N-look_ahead:
            pind = 1

        target = self.wpts[pind]
        self.target = target

        return target, pind


class CarModel:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_dot = 0
        self.y_dot = 0

        self.prev_loc = 0

        self.wheelbase = 0.33
        self.mass = 3.74
        self.mu = 0.523

        # self.max_d_dot = 3.2
        # self.max_steer = 0.4
        # self.max_a = 7.5
        # self.max_decel = -8.5
        self.max_v = 7.5
        self.max_friction_force = self.mass * self.mu * 9.81

    def update_kinematic_state(self, x_dot, y_dot, dt):
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.x = self.x + self.x_dot * dt
        self.y = self.y + self.y_dot * dt

    def get_car_state(self):
        state = []
        state.append(self.x)
        state.append(self.y)
        state.append(self.x_dot)
        state.append(self.y_dot)

        return state


class ScanSimulator:
    def __init__(self, number_of_beams=10, fov=np.pi, std_noise=0.01):
        self.number_of_beams = number_of_beams
        self.fov = fov 
        self.std_noise = std_noise

        self.dth = self.fov / (self.number_of_beams -1)
        self.scan_output = np.zeros(number_of_beams)

        self.step_size = 0.1
        self.n_searches = 50

        self.race_map = None
        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

    def get_scan(self, x, y, theta):
        for i in range(self.number_of_beams):
            scan_theta = theta + self.dth * i - self.fov/2
            self.scan_output[i] = self.trace_ray(x, y, scan_theta)

        return self.scan_output

    def trace_ray(self, x, y, theta, noise=True):
        # obs_res = 10
        for j in range(self.n_searches): # number of search points
            fs = self.step_size * (j + 1)  # search from 1 step away from the point
            dx =  [np.sin(theta) * fs, np.cos(theta) * fs]
            search_val = lib.add_locations([x, y], dx)
            if self._check_location(search_val):
                break       

        ray = (j) / self.n_searches #* (1 + np.random.normal(0, self.std_noise))
        return ray

    def set_map(self, check_fcn):
        self._check_location = check_fcn


class TrackSim:
    def __init__(self, env_map):
        self.timestep = 0.01

        self.env_map = env_map

        self.car = CarModel()
        self.scan_sim = ScanSimulator(10, np.pi*2/3)
        self.scan_sim.set_map(self.env_map.check_scan_location)

        self.done = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.obs_space = len(self.get_state_obs())
        self.ds = 10

        self.steer_history = []
        self.d_ref_history = []
        self.velocity_history = []
        self.done_reason = ""
        self.y_forces = []

    def step(self, action):
        self.steps += 1

        x_dot_ref = action[0]
        y_dot_ref = action[1]
        self.action = action

        frequency_ratio = 10 # cs updates per planning update
        self.car.prev_loc = [self.car.x, self.car.y]
        for i in range(frequency_ratio):
            # acceleration, steer_dot = self.control_system(v_ref, d_ref)
            self.car.update_kinematic_state(x_dot, y_dot, self.timestep)
         
        self.check_done_reward_track_train()

        obs = self.get_state_obs()
        done = self.done
        reward = self.reward

        self.action_memory.append([self.car.x, self.car.y])

        return obs, reward, done, None

    def control_system(self, v_ref, d_ref):

        kp_a = 10
        a = (v_ref - self.car.velocity) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - self.car.steering) * kp_delta

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return a, d_dot

    def reset(self, poses=None, random_start=False):
        self.done = False
        self.action_memory = []
        self.steps = 0
        
        self.car.x = self.env_map.start[0]
        self.car.y = self.env_map.start[1]
        self.car.prev_loc = [self.car.x, self.car.y]
        self.car.velocity = 0
        self.car.steering = 0
        self.car.theta = 0

        return self.get_state_obs()

    def show_history(self):
        plt.figure(3)
        plt.title("Steer history")
        plt.plot(self.steer_history)
        plt.pause(0.001)

        plt.figure(2)
        plt.title("Velocity history")
        plt.plot(self.velocity_history)
        plt.pause(0.001)
        self.velocity_history.clear()

        plt.figure(1)
        plt.title("Forces history")
        plt.plot(self.y_forces)
        plt.pause(0.001)
        self.y_forces.clear()

        plt.figure(5)
        plt.title('D_ref history')
        plt.plot(self.d_ref_history)
        plt.pause(0.0001)
        self.d_ref_history.clear()
        
    def reset_lap(self):
        self.steps = 0
        self.reward = 0
        self.car.prev_loc = [self.car.x, self.car.y]
        self.action_memory.clear()
        self.done = False

    def get_state_obs(self):
        car_state = self.car.get_car_state()
        scan = self.scan_sim.get_scan(self.car.x, self.car.y, self.car.theta)

        state = np.concatenate([car_state, scan])

        return state

    def check_done_reward_track_train(self):
        self.reward = 0 # normal
        if self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"
        horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        self.y_forces.append(horizontal_force)
        if horizontal_force > self.car.max_friction_force:
            self.done = True
            self.reward = -1
            self.done_reason = f"Friction limit reached: {horizontal_force} > {self.car.max_friction_force}"
        if self.steps > 2000:
            self.done = True
            self.done_reason = f"Max steps"
        start_y = self.env_map.start[1]
        if self.car.prev_loc[1] < start_y - 0.5 and self.car.y > start_y - 0.5:
            if abs(self.car.x - self.env_map.start[0]) < 1:
                self.done = True
                self.done_reason = f"Lap complete"

    def render(self, wait=False, wpts=None):
        fig = plt.figure(4)
        plt.clf()  

        c_line = self.env_map.track_pts
        track = self.env_map.track
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        # plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
        plt.plot(l_line[:, 0]*self.ds, l_line[:, 1]*self.ds, linewidth=1)
        plt.plot(r_line[:, 0]*self.ds, r_line[:, 1]*self.ds, linewidth=1)

        ret_map = self.env_map.get_show_map()
        plt.imshow(ret_map.T, origin='lower')

        plt.xlim([0, 100])
        plt.ylim([0, 100])

        plt.plot(self.env_map.start[0]*self.ds, self.env_map.start[1]*self.ds, '*', markersize=12)

        plt.plot(self.env_map.end[0]*self.ds, self.env_map.end[1]*self.ds, '*', markersize=12)
        plt.plot(self.car.x*self.ds, self.car.y*self.ds, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            x = [self.car.x*self.ds, range_val[0]*self.ds]
            y = [self.car.y*self.ds, range_val[1]*self.ds]
            plt.plot(x, y)

        for pos in self.action_memory:
            plt.plot(pos[0]*self.ds, pos[1]*self.ds, 'x', markersize=6)

        if wpts is not None:
            xs, ys = [], []
            for pt in wpts:
                xs.append(pt[0]*self.ds)
                ys.append(pt[1]*self.ds)
        
            # plt.plot(xs, ys)
            plt.plot(xs, ys, 'x', markersize=20)

        if self.env_map.target is not None:
            t_x = self.env_map.target[0] * self.ds
            t_y = self.env_map.target[1] * self.ds
            plt.plot(t_x, t_y, 'x', markersize=18)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(100, 80, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(100, 70, s) 
        s = f"Done: {self.done}"
        plt.text(100, 65, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(100, 60, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(100, 55, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(100, 50, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(100, 45, s) 
        s = f"Theta Dot: [{(self.car.th_dot):.2f}]"
        plt.text(100, 40, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, 35, s)

        plt.pause(0.0001)
        if wait:
            plt.show()
   
    def render_mpc(self, wpts=None, sol=None, wait=False):
        fig = plt.figure(4)
        plt.clf()  

        c_line = self.env_map.track_pts
        track = self.env_map.track
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        # plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
        plt.plot(l_line[:, 0]*self.ds, l_line[:, 1]*self.ds, linewidth=1)
        plt.plot(r_line[:, 0]*self.ds, r_line[:, 1]*self.ds, linewidth=1)

        plt.xlim([0, 100])
        plt.ylim([0, 100])

        plt.plot(self.car.x*self.ds, self.car.y*self.ds, '+', markersize=16)

        for pos in self.action_memory:
            plt.plot(pos[0]*self.ds, pos[1]*self.ds, 'x', markersize=6)

        if wpts is not None:
            xs, ys = [], []
            for pt in wpts:
                xs.append(pt[0]*self.ds)
                ys.append(pt[1]*self.ds)
        
            # plt.plot(xs, ys)
            plt.plot(xs, ys, 'o', markersize=5)

        if sol is not None:
            xs, ys = [], []
            for pt in sol:
                xs.append(pt[0]*self.ds)
                ys.append(pt[1]*self.ds)
        
            # plt.plot(xs, ys)
            plt.plot(xs, ys, 'x', markersize=20)

        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(100, 70, s) 
        s = f"Done: {self.done}"
        plt.text(100, 65, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(100, 60, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(100, 55, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(100, 50, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(100, 45, s) 
        s = f"Theta Dot: [{(self.car.th_dot):.2f}]"
        plt.text(100, 40, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, 35, s)

        plt.pause(0.0001)
        if wait:
            plt.show()
            
    def render_snapshot(self, wait=False, wpts=None):
        fig = plt.figure(4)
        plt.clf()  
        c_line = self.env_map.track_pts
        track = self.env_map.track
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        # plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
        plt.plot(l_line[:, 0]*self.ds, l_line[:, 1]*self.ds, linewidth=1)
        plt.plot(r_line[:, 0]*self.ds, r_line[:, 1]*self.ds, linewidth=1)
        ret_map = self.env_map.get_show_map()
        plt.imshow(ret_map.T, origin='lower')

        plt.xlim([0, 100])
        plt.ylim([0, 100])

        plt.plot(self.env_map.start[0]*self.ds, self.env_map.start[1]*self.ds, '*', markersize=12)

        plt.plot(self.env_map.end[0]*self.ds, self.env_map.end[1]*self.ds, '*', markersize=12)
        plt.plot(self.car.x*self.ds, self.car.y*self.ds, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            x = [self.car.x*self.ds, range_val[0]*self.ds]
            y = [self.car.y*self.ds, range_val[1]*self.ds]
            plt.plot(x, y)

        for pos in self.action_memory:
            plt.plot(pos[0]*self.ds, pos[1]*self.ds, 'x', markersize=6)

        if wpts is not None:
            xs, ys = [], []
            for pt in wpts:
                xs.append(pt[0]*self.ds)
                ys.append(pt[1]*self.ds)
        
            plt.plot(xs, ys)
            # plt.plot(xs, ys, 'x', markersize=20)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(100, 80, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(100, 70, s) 
        s = f"Done: {self.done}"
        plt.text(100, 65, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(100, 60, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(100, 55, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(100, 50, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(100, 45, s) 
        s = f"Done reason: {self.done_reason}"
        plt.text(100, 40, s) 
        

        s = f"Steps: {self.steps}"
        plt.text(100, 35, s)

        plt.pause(0.0001)
        if wait:
            plt.show()




