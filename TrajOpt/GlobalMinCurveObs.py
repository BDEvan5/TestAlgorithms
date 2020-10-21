
import numpy as np
import casadi as ca 
from matplotlib import pyplot as plt 
import yaml, csv

import LibFunctions as lib 



names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'

def MinCurvatureTrajectory(track, obs_map=None):
    w_min = - track[:, 4] * 0.9
    w_max = track[:, 5] * 0.9
    nvecs = track[:, 2:4]
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

    xgrid = np.arange(0, obs_map.shape[1])
    ygrid = np.arange(0, obs_map.shape[0])

    data_flat = np.array(obs_map).ravel(order='F')

    lut = ca.interpolant('lut', 'bspline', [xgrid, ygrid], data_flat)

    print(lut([10, 20]))

    n_max = 3
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # objective
    real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
    im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

    sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N-1)

    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])), 
    # 'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),
                # lut(ca.horzcat(o_x_s(n[:-1]), o_y_s(n[:-1])).T).T,

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    # lbx = [-n_max] * N + [-np.pi]*(N-1) 
    # ubx = [n_max] * N + [np.pi]*(N-1) 
    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    thetas = np.array(x_opt[1*N:2*(N-1)])

    # lib.plot_race_line(np.array(track), n_set, wait=True)

    return n_set


class MapBase:
    def __init__(self, map_name):
        self.name = map_name

        self.scan_map = None
        self.obs_map = None

        self.track = None
        self.track_pts = None
        self.nvecs = None
        self.ws = None
        self.N = None

        self.start = None
        self.wpts = []

        self.height = None
        self.width = None
        self.resolution = None

        self.crop_x = None
        self.crop_y = None

        self.read_yaml_file()
        self.load_map_csv()

    def read_yaml_file(self, print_out=False):
        file_name = 'maps/' + self.name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

            yaml_file = documents.items()
            if print_out:
                for item, doc in yaml_file:
                    print(item, ":", doc)

        self.yaml_file = dict(yaml_file)

        self.resolution = self.yaml_file['resolution']
        self.start = self.yaml_file['start']
        self.crop_x = self.yaml_file['crop_x']
        self.crop_y = self.yaml_file['crop_y']

    def load_map_csv(self):
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
        # self.ws = track[:, 4:6]

        self.scan_map = np.load(f'Maps/{self.name}.npy')
        self.obs_map = np.zeros_like(self.scan_map)

        self.width = self.scan_map.shape[1]
        self.height = self.scan_map.shape[0]

    def render_map(self, figure_n=4, wait=False):
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.width])
        plt.ylim([self.height, 0])

        track = self.track
        c_line = track[:, 0:2]
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        cx, cy = self.convert_positions(c_line)
        plt.plot(cx, cy, linewidth=2)
        lx, ly = self.convert_positions(l_line)
        plt.plot(lx, ly, linewidth=1)
        rx, ry = self.convert_positions(r_line)
        plt.plot(rx, ry, linewidth=1)

        if self.wpts is not None:
            xs, ys = [], []
            for pt in self.wpts:
                x, y = self.convert_position(pt)
                # plt.plot(x, y, '+', markersize=14)
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, '--', linewidth=2)

        if self.obs_map is None:
            plt.imshow(self.scan_map)
        else:
            plt.imshow(self.obs_map + self.scan_map)

        plt.gca().set_aspect('equal', 'datalim')

        plt.pause(0.0001)
        if wait:
            plt.show()

    def convert_position(self, pt):
        x = pt[0] / self.resolution
        y =  pt[1] / self.resolution

        return x, y

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.convert_position(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)
        
    def convert_int_position(self, pt):
        x = int(round(np.clip(pt[0] / self.resolution, 0, self.width-1)))
        y = int(round(np.clip(pt[1] / self.resolution, 0, self.height-1)))

        return x, y

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True

        x, y = self.convert_int_position(x_in)
        if self.scan_map[y, x]:
            return True
        # if self.obs_map[y, x]:
        #     return True
        return False

    def random_obs(self, n=10):
        self.obs_map = np.zeros_like(self.obs_map)

        obs_size = [self.width/600, self.height/600]
        # obs_size = [0.3, 0.3]
        # obs_size = [1, 1]
        x, y = self.convert_int_position(obs_size)
        obs_size = [x, y]
    
        rands = np.random.randint(1, self.N-1, n)
        obs_locs = []
        for i in range(n):
            pt = self.track_pts[rands[i]][:, None]
            obs_locs.append(pt[:, 0])

        for obs in obs_locs:
            for i in range(0, obs_size[0]):
                for j in range(0, obs_size[1]):
                    x, y = self.convert_int_position([obs[0], obs[1]])
                    self.obs_map[y+j, x+i] = 1

    def set_true_widths(self):
        nvecs = self.track[:, 2:4]
        tx = self.track[:, 0]
        ty = self.track[:, 1]

        stp_sze = 0.1
        sf = 0.9 # safety factor
        nws, pws = [], []
        for i in range(self.N):
            pt = [tx[i], ty[i]]
            nvec = nvecs[i]

            j = stp_sze
            s_pt = s_pt = lib.add_locations(pt, nvec, j)
            while not self.check_scan_location(s_pt):
                j += stp_sze
                s_pt = lib.add_locations(pt, nvec, j)
            pws.append(j*sf)

            j = stp_sze
            s_pt = s_pt = lib.sub_locations(pt, nvec, j)
            while not self.check_scan_location(s_pt):
                j += stp_sze
                s_pt = lib.sub_locations(pt, nvec, j)
            nws.append(j*sf)

        nws, pws = np.array(nws), np.array(pws)
        self.ws = np.concatenate([nws[:, None], pws[:, None]])

        new_track = np.concatenate([self.track[:, 0:4], nws[:, None], pws[:, None]], axis=-1)

        self.track = new_track


    def get_min_curve(self):
        track = self.track
        n_set = MinCurvatureTrajectory(track, self.obs_map)
        deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
        path = track[:, 0:2] + deviation
        self.wpts = path


def main():
    mymap = MapBase(name)
    mymap.set_true_widths()
    mymap.random_obs(10)
    mymap.get_min_curve()
    mymap.render_map(wait=True)






if __name__ == "__main__":
    main()