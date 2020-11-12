import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt




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


def BaseOpti():
    pathpoints = 30
    ref_path = {}
    ref_path['x'] = 5*np.sin(np.linspace(0,2*np.pi, pathpoints+1))
    ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
    wp = ca.horzcat(ref_path['x'], ref_path['y']).T

    N = 5
    N1 = N-1

    wpts = np.array(wp[:, 0:N])

    x = ca.MX.sym('x', N)
    y = ca.MX.sym('y', N)
    x_dot = ca.MX.sym('xd', N-1)
    y_dot = ca.MX.sym('yd', N-1)
    dt = ca.MX.sym('dt', N-1)

    nlp = {\
        'x': ca.vertcat(x, y, x_dot, y_dot, dt),
        'f': ca.sumsqr(x - wpts[0, :]) + ca.sumsqr(y - wpts[1, :]) + ca.sumsqr(dt),
        'g': ca.vertcat(\
            x[1:] - (x[:-1] + x_dot * dt),
            y[1:] - (y[:-1] + y_dot * dt),

            x[0] - wpts[0, 0], 
            y[0] - wpts[1, :],
            x_dot[0], 
            y_dot[0]
            ),
        }
        
    S = ca.nlpsol('S', 'ipopt', nlp)

    # x0
    x00 = wpts[0, :]
    y00 = wpts[1, :]
    T = 5
    dt00 = [T/N1] * N1
    xd00 = (x00[1:] - x00[:-1]) / dt00
    yd00 = (y00[1:] - y00[:-1]) / dt00

    x0 = ca.vertcat(x00, y00, xd00, yd00, dt00)

    max_speed = 1

    lbx = [0] * 2 * N + [-max_speed] * 2 * N1 + [0] * N1
    ubx = [ca.inf] * 2 * N + [max_speed] * 2 * N1 + [10] * N1

    # solve
    r = S(x0=x0, lbx=lbx, ubx=ubx)
    x_opt = r['x']

    x = np.array(x_opt[:N])
    y = np.array(x_opt[N:2*N])
    times = np.array(x_opt[N*2 + N1*2:])
    xds = np.array(x_opt[2*N:2*N + N1])
    yds = np.array(x_opt[2*N + N1: 2*N + 2*N1])
    total_time = np.sum(times)

    print(f"Times: {times}")
    print(f"Total Time: {total_time}")
    print(f"X dots: {xds.T}")
    print(f"Y dots: {yds.T}")
    print(f"xs: {x.T}")
    print(f"ys: {y.T}")

    plt.figure(1)
    plt.plot(wpts[0, :], wpts[1, :], 'x')

    plt.plot(x, y, '+', markersize=2)

    plt.show()


if __name__ == "__main__":
    BaseOpti()
