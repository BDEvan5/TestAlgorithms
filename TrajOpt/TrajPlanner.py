import numpy as np 
import matplotlib.pyplot as plt 
import csv
import math
import casadi as ca

import os
import sys
import time

import LibFunctions as lib 
# from RaceTrackMaps import RaceMap
# from PathFinder import PathFinder


def create_track():
    track_name = 'RaceTrack1000'
    env_map = RaceMap('RaceTrack1000')
    env_map.reset_map()
    fcn = env_map.obs_free_hm._check_line
    path_finder = PathFinder(fcn, env_map.start, env_map.end)

    path = path_finder.run_search(5)
    # env_map.race_course.show_map(path=path, show=True)

    width = 5
    widths = np.ones_like(path) * width
    track = np.concatenate([path, widths], axis=-1)

    filename = 'Maps/' + track_name + '_abscissa.csv'

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(['xi', 'yi', 'nl', 'nr'])
        csvwriter.writerows(track)

    print(f"Track Created")


def load_track(filename='Maps/RaceTrack1000_abscissa.csv', show=True):
    track = []
    with open(filename, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
    
        for lines in csvFile:  
            track.append(lines)

    track = np.array(track)

    print(f"Track Loaded")

    if show:
        plot_track(track)

    return track

def interp_track(track, step_size=1):
    ref_track_closed = np.vstack((track, track[0]))

    seg_lengths = np.sqrt(np.sum(np.power(np.diff(ref_track_closed[:, :2], axis=0), 2), axis=1))
    dists_cum = np.cumsum(seg_lengths)
    dists_cum = np.insert(dists_cum, 0, 0.0)

    no_points_interp = math.ceil(dists_cum[-1] / step_size) + 1
    dists_interp = np.linspace(0.0, dists_cum[-1], no_points_interp)

    interp_track = np.zeros((no_points_interp, 4))
    interp_track[:, 0] = np.interp(dists_interp, dists_cum, ref_track_closed[:, 0])
    interp_track[:, 1] = np.interp(dists_interp, dists_cum, ref_track_closed[:, 1])
    interp_track[:, 2] = np.interp(dists_interp, dists_cum, ref_track_closed[:, 2])
    interp_track[:, 3] = np.interp(dists_interp, dists_cum, ref_track_closed[:, 3])

    return interp_track

def generate_bounds(track:np.ndarray, normvec):
    lx = track[:, 0] - track[:, 2] * normvec[:, 0]
    ly = track[:, 1] - track[:, 2] * normvec[:, 1]
    rx = track[:, 0] + track[:, 3] * normvec[:, 0]
    ry = track[:, 1] + track[:, 3] * normvec[:, 1]

    l = np.concatenate((lx[:, None], ly[:, None]), axis=-1)
    r = np.concatenate((rx[:, None], ry[:, None]), axis=-1)
    
    return l, r

def generate_raceline(track:np.ndarray, n_set, normvec):
    lx = track[:, 0] + n_set * normvec[:, 0]
    ly = track[:, 1] + n_set * normvec[:, 1]

    l = np.concatenate((lx[:, None], ly[:, None]), axis=-1)
    
    return l

def calc_splines(path):
    path = np.vstack((path, path[0, :]))
    seg_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))

    no_splines = path.shape[0] - 1

    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))

    scaling = np.ones(no_splines - 1)

    template_M = np.array(                          # current point               | next point              | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0

    for i in range(no_splines):
        j = i * 4

        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M

            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)

        else:
            # no curvature and heading bounds on last element (handled afterwards)
            M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],
                                     [1,  1,  1,  1]]

        b_x[j: j + 2] = [[path[i,     0]],
                         [path[i + 1, 0]]]
        b_y[j: j + 2] = [[path[i,     1]],
                         [path[i + 1, 1]]]

    M[-2, 1] = scaling[-1]
    M[-2, -3:] = [-1, -2, -3]
    # b_x[-2] = 0
    # b_y[-2] = 0

    # curvature boundary condition (for a closed spline)
    M[-1, 2] = 2 * math.pow(scaling[-1], 2)
    M[-1, -2:] = [-2, -6]
    # b_x[-1] = 0
    # b_y[-1] = 0

    x_les = np.squeeze(np.linalg.solve(M, b_x))  # squeeze removes single-dimensional entries
    y_les = np.squeeze(np.linalg.solve(M, b_y))

    # get coefficients of every piece into one row -> reshape
    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))

    # get normal vector (behind used here instead of ahead for consistency with other functions) (second coefficient of
    # cubic splines is relevant for the heading)
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

    # normalize normal vectors
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    normvec_normalized = np.expand_dims(norm_factors, axis=1) * normvec

    return coeffs_x, coeffs_y, normvec_normalized



def plot_track(track):
    track = np.vstack((track, track[-1, :]))

    x, y, normvec = calc_splines(track)
    l, r = generate_bounds(track, normvec)

    plt.figure()

    plt.plot(track[:, 0], track[:, 1], linewidth=1)
    plt.plot(l[:, 0], l[:, 1], linewidth=2)
    plt.plot(r[:, 0], r[:, 1], linewidth=2)

    plt.pause(0.001)
    plt.show()

def plot_race_line(track, n_set):
    track = np.vstack((track, track[-1, :]))

    x, y, normvec = calc_splines(track)
    l, r = generate_bounds(track, normvec)

    plt.figure()

    plt.plot(track[:, 0], track[:, 1], linewidth=1)
    plt.plot(l[:, 0], l[:, 1], linewidth=1)
    plt.plot(r[:, 0], r[:, 1], linewidth=1)

    line = generate_raceline(track, n_set, normvec)
    plt.plot(line[:, 0], line[:, 1], linewidth=2)

    plt.pause(0.001)
    plt.show()


"""Optimisation stufff"""
def calc_spline_lengths(coeffs_x, coeffs_y):
    no_splines = coeffs_x.shape[0]
    spline_lengths = np.zeros(no_splines)
    no_interp_points = 15

    t_steps = np.linspace(0.0, 1.0, no_interp_points)
    spl_coords = np.zeros((no_interp_points, 2))

    for i in range(no_splines):
        spl_coords[:, 0] = coeffs_x[i, 0] \
                            + coeffs_x[i, 1] * t_steps \
                            + coeffs_x[i, 2] * np.power(t_steps, 2) \
                            + coeffs_x[i, 3] * np.power(t_steps, 3)
        spl_coords[:, 1] = coeffs_y[i, 0] \
                            + coeffs_y[i, 1] * t_steps \
                            + coeffs_y[i, 2] * np.power(t_steps, 2) \
                            + coeffs_y[i, 3] * np.power(t_steps, 3)

        spline_lengths[i] = np.sum(np.sqrt(np.sum(np.power(np.diff(spl_coords, axis=0), 2), axis=1)))

    return spline_lengths

def normalize_psi(psi):
    psi_out = np.sign(psi) * np.mod(np.abs(psi), 2 * math.pi)

    # restrict psi to [-pi,pi]
    if type(psi_out) is np.ndarray:
        psi_out[psi_out >= math.pi] -= 2 * math.pi
        psi_out[psi_out < -math.pi] += 2 * math.pi

    else:
        if psi_out >= math.pi:
            psi_out -= 2 * math.pi
        elif psi_out < -math.pi:
            psi_out += 2 * math.pi

    return psi_out

def calc_head_curve(path, seg_lengths):
    no_points = path.shape[0]
    stepsize_psi_preview = 1.0
    stepsize_psi_review = 1.0
    stepsize_curv_preview = 2.0
    stepsize_curv_review = 2.0

    # prework
    ind_step_preview_psi = round(stepsize_psi_preview / float(np.average(seg_lengths)))
    ind_step_review_psi = round(stepsize_psi_review / float(np.average(seg_lengths)))
    ind_step_preview_curv = round(stepsize_curv_preview / float(np.average(seg_lengths)))
    ind_step_review_curv = round(stepsize_curv_review / float(np.average(seg_lengths)))

    ind_step_preview_psi = max(ind_step_preview_psi, 1)
    ind_step_review_psi = max(ind_step_review_psi, 1)
    ind_step_preview_curv = max(ind_step_preview_curv, 1)
    ind_step_review_curv = max(ind_step_review_curv, 1)

    steps_tot_psi = ind_step_preview_psi + ind_step_review_psi
    steps_tot_curv = ind_step_preview_curv + ind_step_review_curv

    # heading
    path_temp = np.vstack((path[-ind_step_review_psi:], path, path[:ind_step_preview_psi]))
    tangvecs = np.stack((path_temp[steps_tot_psi:, 0] - path_temp[:-steps_tot_psi, 0],
                            path_temp[steps_tot_psi:, 1] - path_temp[:-steps_tot_psi, 1]), axis=1)

    # calculate psi of tangent vectors (pi/2 must be substracted due to our convention that psi = 0 is north)
    psi = np.arctan2(tangvecs[:, 1], tangvecs[:, 0]) - math.pi / 2
    psi = normalize_psi(psi)

    # curvature
    psi_temp = np.insert(psi, 0, psi[-ind_step_review_curv:])
    psi_temp = np.append(psi_temp, psi[:ind_step_preview_curv])

    # calculate delta psi
    delta_psi = np.zeros(no_points)

    for i in range(no_points):
        delta_psi[i] = normalize_psi(psi_temp[i + steps_tot_curv] - psi_temp[i])

    # calculate kappa
    s_points_cl = np.cumsum(seg_lengths)
    s_points_cl = np.insert(s_points_cl, 0, 0.0)
    s_points = s_points_cl[:-1]
    s_points_cl_reverse = np.flipud(-np.cumsum(np.flipud(seg_lengths)))  # should not include 0.0 as last value

    s_points_temp = np.insert(s_points, 0, s_points_cl_reverse[-ind_step_review_curv:])
    s_points_temp = np.append(s_points_temp, s_points_cl[-1] + s_points[:ind_step_preview_curv])

    delta_psi = delta_psi[:-1]
    kappa = delta_psi / (s_points_temp[steps_tot_curv:] - s_points_temp[:-steps_tot_curv])

    return psi, kappa

def opt_mintime(reftrack, c_x, c_y, normvecs):
    no_points_orig = reftrack.shape[0]
    discr_points = np.arange(reftrack.shape[0])
    reftrack = np.vstack((reftrack, reftrack[0, :]))

    spline_lengths = calc_spline_lengths(c_x, c_y)

    # track heading - closed
    psi, kappa_refline = calc_head_curve(reftrack[:, :2], spline_lengths)

    # close track
    kappa_refline_cl = np.append(kappa_refline, kappa_refline[0])
    discr_points_cl = np.append(discr_points, no_points_orig)  # add virtual index of last/first point for closed track
    w_tr_left_cl = np.append(reftrack[:, 3], reftrack[0, 3])
    w_tr_right_cl = np.append(reftrack[:, 2], reftrack[0, 2])

    h = 3
    steps = [i for i in range(discr_points_cl.size)]
    N = steps[-1]
    s_opt = np.asarray(discr_points_cl) * h

    kappa_interp = ca.interpolant('kappa_interp', 'linear', [steps], kappa_refline_cl)
    w_tr_left_interp = ca.interpolant('w_tr_left_interp', 'linear', [steps], w_tr_left_cl)
    w_tr_right_interp = ca.interpolant('w_tr_right_interp', 'linear', [steps], w_tr_right_cl)

"""For later"""
     # ------------------------------------------------------------------------------------------------------------------
    # DIRECT GAUSS-LEGENDRE COLLOCATION --------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # # degree of interpolating polynomial
    # d = 3

    # # legendre collocation points
    # tau = np.append(0, ca.collocation_points(d, 'legendre'))

    # # coefficient matrix for formulating the collocation equation
    # C = np.zeros((d + 1, d + 1))

    # # coefficient matrix for formulating the collocation equation
    # D = np.zeros(d + 1)

    # # coefficient matrix for formulating the collocation equation
    # B = np.zeros(d + 1)

    # # construct polynomial basis
    # for j in range(d + 1):
    #     # construct Lagrange polynomials to get the polynomial basis at the collocation point
    #     p = np.poly1d([1])
    #     for r in range(d + 1):
    #         if r != j:
    #             p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

    #     # evaluate polynomial at the final time to get the coefficients of the continuity equation
    #     D[j] = p(1.0)

    #     # evaluate time derivative of polynomial at collocation points to get the coefficients of continuity equation
    #     p_der = np.polyder(p)
    #     for r in range(d + 1):
    #         C[j, r] = p_der(tau[r])

    #     # evaluate integral of the polynomial to get the coefficients of the quadrature function
    #     pint = np.polyint(p)
    #     B[j] = pint(1.0)

    # # state variables
    # nx = 5
    # nx_pwr = 0

    # # velocity [m/s]
    # v_n = ca.SX.sym('v_n')
    # v_s = 5
    # v = v_s * v_n

    # # side slip angle [rad]
    # beta_n = ca.SX.sym('beta_n')
    # beta_s = 0.5
    # beta = beta_s * beta_n

    # # yaw rate [rad/s]
    # omega_z_n = ca.SX.sym('omega_z_n')
    # omega_z_s = 1
    # omega_z = omega_z_s * omega_z_n

    # # lateral distance to reference line (positive = left) [m]
    # n_n = ca.SX.sym('n_n')
    # n_s = 1
    # n = n_s * n_n

    # # relative angle to tangent on reference line [rad]
    # xi_n = ca.SX.sym('xi_n')
    # xi_s = 1.0
    # xi = xi_s * xi_n

    # # ------------------------------------------------------------------------------------------------------------------
    # # CONTROL VARIABLES ------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------

    # # number of control variables
    # nu = 4

    # # steer angle [rad]
    # delta_n = ca.SX.sym('delta_n')
    # delta_s = 0.5
    # delta = delta_s * delta_n

    # # positive longitudinal force (drive) [N]
    # f_drive_n = ca.SX.sym('f_drive_n')
    # f_drive_s = 7500.0
    # f_drive = f_drive_s * f_drive_n

    # # negative longitudinal force (brake) [N]
    # f_brake_n = ca.SX.sym('f_brake_n')
    # f_brake_s = 20000.0
    # f_brake = f_brake_s * f_brake_n

    # # lateral wheel load transfer [N]
    # gamma_y_n = ca.SX.sym('gamma_y_n')
    # gamma_y_s = 5000.0
    # gamma_y = gamma_y_s * gamma_y_n

    # g = 9.81
    
    

    # # scaling factors for control variables
    # u_s = np.array([delta_s, f_drive_s, f_brake_s, gamma_y_s])

    # # put all controls together
    # u = ca.vertcat(delta_n, f_drive_n, f_brake_n, gamma_y_n)

def opt_mindist(reftrack, c_x, c_y, normvecs):
    no_points_orig = reftrack.shape[0]
    discr_points = np.arange(reftrack.shape[0])
    reftrack = np.vstack((reftrack, reftrack[0, :]))

    spline_lengths = calc_spline_lengths(c_x, c_y)

    # track heading - closed
    psi, kappa_refline = calc_head_curve(reftrack[:, :2], spline_lengths)

    # close track
    kappa_refline_cl = np.append(kappa_refline, kappa_refline[0])
    discr_points_cl = np.append(discr_points, no_points_orig) 

    





if __name__ == "__main__":
    # create_track()
    track = load_track(show=False)

    track = interp_track(track)

    x, y, nvec = calc_splines(track)

    plot_track(track)

    opt_mintime(track, x, y, nvec)




