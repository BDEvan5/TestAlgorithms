import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt

from MapGen import run_map_gen, plot_race_line
# from TrajPlanner import plot_race_line
import LibFunctions as lib

def MinDistance():
    track = run_map_gen()
    txs = track[:, 0]
    tys = track[:, 1]
    nvecs = track[:, 2:4]

    n_max = 5
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    grad = ca.Function('grad', [n_f_a], [(o_y_e(n_f_a[1:]) - o_y_s(n_f_a[:-1]))/ca.fmax(o_x_e(n_f_a[1:]) - o_x_s(n_f_a[:-1]), 0.1*np.ones(N-1) )])
    curvature = ca.Function('curv', [n_f_a], [grad(n_f_a)[1:] - grad(n_f_a)[:-1]]) # changes in grad

    # define symbols
    n = ca.MX.sym('n', N)

    nlp = {\
    'x': ca.vertcat(n),
    'f': ca.sumsqr(track_length(n)),
    # 'f': ca.sumsqr(curvature(n)), 
    'g': ca.vertcat(
                # boundary constraints
                n[0], 
                n[-1], 
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp)

    ones = np.ones(N)
    n0 = ones*0

    x0 = ca.vertcat(n0)

    lbx = [-n_max] * N
    ubx = [n_max] * N

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    print(f"Solution found")
    x_opt = r['x']

    n_set = np.array(x_opt[:N])

    plot_race_line(np.array(track), n_set, width=5, wait=True)
    
def MinCurvature():
    track = run_map_gen()
    txs = track[:, 0]
    tys = track[:, 1]
    nvecs = track[:, 2:4]
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]
    # sls = np.sqrt(np.sum(np.power(np.diff(track[:, :2], axis=0), 2), axis=1))


    n_max = 3
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f_a = ca.MX.sym('n_f', N)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan(im(th1_f, th2_f)/real(th1_f, th2_f))])
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])

    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])
    # curvature = ca.Function('curv', [th_f_a], [th_f_a[1:] - th_f_a[:-1]])
    grad = ca.Function('grad', [n_f_a], [(o_y_e(n_f_a[1:]) - o_y_s(n_f_a[:-1]))/ca.fmax(o_x_e(n_f_a[1:]) - o_x_s(n_f_a[:-1]), 0.1*np.ones(N-1) )])
    curvature = ca.Function('curv', [n_f_a], [grad(n_f_a)[1:] - grad(n_f_a)[:-1]]) # changes in grad
   

    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N)


    nlp = {\
    'x': ca.vertcat(n, th),
    # 'f': ca.sumsqr(curvature(n)),
    'f': ca.sumsqr(sub_cmplx(th[1:], th[:-1])), 
    # 'f': ca.sumsqr(track_length(n)),
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th[:-1])),

                # boundary constraints
                n[0], th[0],
                n[-1], #th[-1],
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp)

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        th0.append(th_00)

    th0.append(0)
    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = [-n_max] * N + [-np.pi]*N 
    ubx = [n_max] * N + [np.pi]*N 

    print(curvature(n0))

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    print(f"Solution found")
    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    thetas = np.array(x_opt[1*N:2*N])

    # print(sub_cmplx(thetas[1:], thetas[:-1]))

    plot_race_line(np.array(track), n_set, width=3, wait=True)
    
def MinCurvatureSteer():
    track = run_map_gen()
    txs = track[:, 0]
    tys = track[:, 1]
    nvecs = track[:, 2:4]
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]
    sls = np.sqrt(np.sum(np.power(np.diff(track[:, :2], axis=0), 2), axis=1))


    l = 0.33
    a_max = 7.5
    d_max = 0.2
    v_max = 6
    n_max = 3
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    d_f = ca.MX.sym('d_f', N-1)
    th_f = ca.MX.sym('th_f', N-1)
    v_f = ca.MX.sym('v_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('th_f', N-1)
    th2_f = ca.MX.sym('th_f', N-1)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    # d_n = ca.Function('d_n', [n_f, th_f], [sls/ca.tan(th_ns[:-1] - th_f)])
    # sls_f = ca.Function('sls_f', [n_f_a], )
    # get_th = ca.Function('gth', [th_f], [th_ns[:-1] - th_f])
    # get_th_n = ca.Function('gth', [th_f], [th_f - th_ns[:-1] + (np.pi/2)*np.ones(N-1)])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan(im(th1_f, th2_f)/real(th1_f, th2_f))])
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(th_f, th_ns[:-1])])
    
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])
    d_t = ca.Function('d_t', [n_f_a, v_f], [dis(o_x_e(n_f_a[1:]), o_x_s(n_f_a[:-1]), o_y_e(n_f_a[1:]), o_y_s(n_f_a[:-1]))/v_f ])
    d_th = ca.Function('d_th', [v_f, d_f], [v_f/l * ca.tan(d_f)])



    grad = ca.Function('grad', [n_f_a], [(o_y_e(n_f_a[1:]) - o_y_s(n_f_a[:-1]))/ca.fmax(o_x_e(n_f_a[1:]) - o_x_s(n_f_a[:-1]), 0.01*np.ones(N-1) )])
    curvature = ca.Function('curv', [n_f_a], [grad(n_f_a)[1:] - grad(n_f_a)[:-1]]) # changes in grad
   

    # define symbols
    n = ca.MX.sym('n', N)
    dt = ca.MX.sym('t', N)
    v = ca.MX.sym('v', N)
    th = ca.MX.sym('th', N)
    a = ca.MX.sym('a', N)
    d = ca.MX.sym('d', N)


    nlp = {\
    'x': ca.vertcat(n, dt, v, th, a, d),
    # 'f': ca.sumsqr(curvature(n)),
    # 'f': ca.sumsqr(track_length(n)),
    # 'f': ca.sumsqr(d) - ca.sumsqr(v),
    'f':  - ca.sumsqr(v),
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th[:-1])),
                th[1:] - (th[:-1] + d_th(v[:-1], d[:-1]) * (dt[:-1])),
                v[1:] - (v[:-1] + a[:-1] * dt[:-1]),
                dt[:-1] - (d_t(n, v[:-1])),

                # boundary constraints
                n[0], d[0], v[0] - 1, 
                n[-1], th[-1], d[-1], dt[-1]
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp)

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        th0.append(th_00)

    th0.append(0)
    th0 = np.array(th0)

    d0 = []
    for i in range(N-1):
        angle = lib.sub_angles_complex(th0[i+1], th0[i])
        d00 = ca.atan(l * (angle) / 3)
        d0.append(d00)
    d0.append(0)
    d0 = np.array(d0)

    v0, a0 = [1], []
    for i in range(N-1):
        if v0[-1] < 3:
            a0.append(0.2)
        else:
            a0.append(0)
        new_v = np.sqrt(v0[-1]**2 + 2 *a0[-1] * sls[i])
        v0.append(new_v)
    a0.append(0) # last a

    t0 = [0]
    for i in range(N-1):
        new_t =  sls[i] / v0[i] # t0[-1] 
        t0.append(new_t)


    x0 = ca.vertcat(n0, t0, v0, th0, a0, d0)

    lbx = [-n_max]*N + [0] * N + [0] * N + [-np.pi]*N + [-a_max] *N + [-d_max] * N
    ubx = [n_max]*N + [np.inf] * N + [v_max] * N + [np.pi]*N + [a_max] *N + [d_max] * N

    # plot_x_opt(x0, track)

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    print(f"Solution found")
    x_opt = r['x']

    plt.figure(7)
    plt.plot(track_length(x_opt[:N]))

    plot_x_opt(x_opt, track)

def plot_x_opt(x_opt, track):
    N = len(track)
    n_set = np.array(x_opt[:N])
    times = np.array(x_opt[N:2*N])
    velocities = np.array(x_opt[2*N:3*N])
    thetas = np.array(x_opt[3*N:4*N])
    accs = np.array(x_opt[4*N:5*N])
    deltas = np.array(x_opt[5*N:6*N])

    plot_race_line(np.array(track), n_set, width=3, wait=False)

    plt.figure(2)
    plt.title('Velocities and As')
    plt.plot(velocities)
    plt.plot(accs)

    plt.pause(0.001)

    plt.figure(3)
    plt.title('Thetas')
    plt.plot(thetas)
    plt.plot(thetas, 'x', markersize=12)
    

    plt.pause(0.0001)

    plt.figure(4)
    plt.title('Deltas')
    plt.plot(deltas)

    plt.pause(0.001)

    plt.figure(5)
    plt.title('N set')
    plt.plot(n_set, 'x')
    plt.plot(n_set)

    plt.figure(6)
    plt.title('d theta')
    # plt.plot(thetas[1:] - thetas[:-1])
    d_ths = [lib.sub_angles_complex(thetas[i+1], thetas[i]) for i in range(N-1)]
    d_ths.append(0)
    plt.plot(d_ths)

    # p_ths = np.concatenate([thetas, np.array(d_ths)[:, None]], axis=-1)
    # print(f"Thetas: {p_ths}")

    plt.show()



if __name__ == "__main__":
    # MinDistance()
    MinCurvature()
    # MinCurvatureSteer()