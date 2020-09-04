import numpy as np
from matplotlib import  pyplot as plt


def add_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret


def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)
     
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret

def get_gradient(x1=[0, 0], x2=[0, 0]):
    t = (x1[1] - x2[1])
    b = (x1[0] - x2[0])
    if b != 0:
        return t / b
    return 1000000 # near infinite gradient. 

def get_angle(x1, x2, x3):
    m1 = get_gradient(x1, x2)
    m2 = get_gradient(x2, x3)
    angle = abs(np.arctan(m1) - np.arctan(m2))
    return angle

def transform_coords(x=[0, 0], theta=np.pi):
    # i want this function to transform coords from one coord system to another
    new_x = x[0] * np.cos(theta) - x[1] * np.sin(theta)
    new_y = x[0] * np.sin(theta) + x[1] * np.cos(theta)

    return np.array([new_x, new_y])

def normalise_coords(x=[0, 0]):
    r = x[0]/x[1]
    y = np.sqrt(1/(1+r**2)) * abs(x[1]) / x[1] # carries the sign
    x = y * r
    return [x, y]

def get_bearing(x1=[0, 0], x2=[0, 0]):
    grad = get_gradient(x1, x2)
    dx = x2[0] - x1[0]
    th_start_end = np.arctan(grad)
    if th_start_end > 0:
        if dx >= 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi/2 - th_start_end
    else:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = - np.pi/2 - th_start_end

    return th_start_end

def theta_to_xy(theta):
    x = np.sin(theta)
    y = np.cos(theta)

    return [x, y]

def get_rands(a=100, b=0):
    r = [np.random.random() * a + b, np.random.random() * a + b]
    return r

def get_rand_ints(a=100, b=0):
    r = [int(np.random.random() * a + b), int(np.random.random() * a + b)]
    return r

def limit_theta(theta):
    if theta > np.pi:
        theta = theta - 2*np.pi
    elif theta < -np.pi:
        theta += 2*np.pi

    return theta


def plot(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    moving_avg = get_moving_average(moving_avg_period * 5, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)

def plot_no_avg(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    plt.pause(0.0001)

def plot_comp(values1, values2,  moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values1)

    moving_avg = get_moving_average(moving_avg_period, values1)
    # plt.plot(moving_avg)    

    plt.plot(values2)
    moving_avg = get_moving_average(moving_avg_period, values2)
    # plt.plot(moving_avg)    
 
    plt.legend(['RL Moving Avg', "Classical Moving Avg"])
    # plt.legend(['RL Agent', 'RL Moving Avg', 'Classical Agent', "Classical Moving Avg"])
    plt.pause(0.001)

def plot_three(values1, values2, values3, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    plt.ylim(-2, 2)

    plt.plot(values1)
    # moving_avg = get_moving_average(moving_avg_period, values1)
    # plt.plot(moving_avg)    

    plt.plot(values2)
    # moving_avg = get_moving_average(moving_avg_period, values2)
    # plt.plot(moving_avg)    

    plt.plot(values3)
    # moving_avg = get_moving_average(moving_avg_period, values2)
    # plt.plot(moving_avg)    
 
    # plt.legend(['RL Moving Avg', "Classical Moving Avg"])
    # plt.legend(['RL Agent', 'RL Moving Avg', 'Classical Agent', "Classical Moving Avg"])
    plt.legend(['Values', 'Q_vals', 'Loss'])
    plt.pause(0.001)

def get_moving_average(period, values):

    moving_avg = np.zeros_like(values)

    for i, avg in enumerate(moving_avg):
        if i > period:
            moving_avg[i] = np.mean(values[i-period:i])
        # else already zero
    return moving_avg

def plot_multi(value_array, title="Results", figure_n=2, ylim=[-1, 1]):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    value_array = np.array(value_array)

    plt.ylim(ylim)

    n_sets = len(value_array[0])
    leg = []
    for i in range(n_sets):
        plt.plot(value_array[:, i])
        leg.append(f"{i}")
    
    plt.legend(leg)

    # plt.plot(values)
    plt.pause(0.0001)


if __name__ == "__main__":
    pass