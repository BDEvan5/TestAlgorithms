import LibFunctions as f 
import numpy as np 
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from scipy import interpolate


class PathFinder:
    def __init__(self, o_function, start, end):
        # ds is the search size around the current node
        self.ds = None
        self.obstacle_function = o_function

        self.open_list = []
        self.closed_list = []
        self.children = []
        
        self.position_list = []
        self.current_node = Node()
        self.open_node_n = 0

        self.start = start
        self.end = end

    def set_directions(self):
        for i in range(3):
            for j in range(3):
                direction = [(j-1)*self.ds, (i-1)*self.ds]
                self.position_list.append(direction)

        self.position_list.pop(4) # remove stand still

        # this makes it not go diagonal
        self.straight_list = [[self.ds, 0], [0, -self.ds], [-self.ds, 0], [0, self.ds]]
        for pos in self.position_list:
            pos[0] = pos[0] #* self.ds
            pos[1] = pos[1] #* self.ds
        # print(self.position_list)

    def run_search(self, ds, max_steps=4000):
        self.ds = ds
        self.set_directions()
        self.set_up_start_node()
        i = 0
        while len(self.open_list) > 0 and i < max_steps:
            self.find_current_node()

            if self.check_done():
                # print("The shortest path has been found")
                break
            self.generate_children()
            i += 1
            # self.plot_nodes()

            if i % 100 == 0:
                print(f"Search step: {i}")
        # print(f"Number of itterations: {i}")
        assert i < max_steps, "Max Iterations reached: problem with search"
        assert len(self.open_list) > 0, "Search broke: no open nodes"
        path = self.get_path_list()

        return path

    def set_up_start_node(self):
        self.start_n = Node(None, np.array(self.start))
        self.end_n = Node(None, np.array(self.end))

        self.open_list.append(self.start_n)

    def find_current_node(self):
        self.current_node = self.open_list[0]
        current_index = 0
        for index, item in enumerate(self.open_list):
            if item.f < self.current_node.f:
                self.current_node = item
                current_index = index
        # Pop current off open list, add to closed list
        self.open_list.pop(current_index)
        self.closed_list.append(self.current_node)
        # self.logger.debug(self.current_node.log_msg())

    def check_done(self):
        dx_max = self.ds * 1.2
        dis = f.get_distance(self.current_node.position, self.end_n.position)
        if dis < dx_max:
            # print("Found")
            return True
        return False

    def _check_closed_list(self, new_node):
        for closed_child in self.closed_list:
            if (new_node == closed_child).all():
                return True
        return False

    def _check_open_list(self, new_node):
        for open_node in self.open_list:
            if (new_node == open_node).all(): # if in open set return true
                if new_node.g < open_node.g:
                    open_node.g = new_node.g
                    open_node.parent = new_node.parent
                return True
        return False

    def generate_children(self):
        self.children.clear() # deletes old children

        for direction in self.position_list:
            new_position = f.add_locations(self.current_node.position, direction)

            if self.obstacle_function(new_position, self.current_node.position): 
                continue 
            new_node = Node(self.current_node, np.array(new_position))

            if self._check_closed_list(new_node): 
                continue
           
            # Create the f, g, and h values
            # takes if it is straight or not into cost consideration.
            if direction in self.straight_list:
                new_node.g = self.current_node.g + self.ds
            else:
                new_node.g = self.current_node.g + self.ds * np.sqrt(2)
            # new_node.g = self.current_node.g + self.ds
            h_val = f.get_distance(new_node.position, self.end_n.position)
            new_node.h = h_val 
            new_node.f = new_node.g + new_node.h

             # Child is already in the open list
            if self._check_open_list(new_node):
                # if 
                continue

            self.open_list.append(new_node)
            self.open_node_n += 1
            # self.plot_nodes()

    def get_path_list(self):
        curr = self.current_node
        pos_list = []
        while curr is not None:
            pos_list.append(curr.position)
            curr = curr.parent
        pos_list = pos_list[::-1]

        pos_list.append(self.end)

        return pos_list

    def plot_nodes(self):
        plt.figure(2)
        plt.clf()
        for node in self.closed_list:
            plt.plot(node.position[0], node.position[1], 'x')
        for node in self.open_list:
            plt.plot(node.position[0], node.position[1], '+')

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        plt.pause(0.001)


class Node():
    """A node class for A* Pathfinding"""
    # helper class for a star

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def log_msg(self):
        if self.parent is not None:
            msg1 = "Parent: " + str(self.parent.position)
            msg2 = "; Pos: " + str(self.position)
            msg3 = "; G: %d -> h: %d -> f: %d " %(self.g, self.h, self.f)

            msg = msg1 + msg2 + msg3
            return msg

 



def reduce_path(path):
    new_path = []
    new_path.append(path[0]) # starting pos
    pt1 = path[0]
    for i in range(2, len(path)):
        pt2 = path[i]
        if pt1[0] != pt2[0] and pt1[1] != pt2[1]:
            new_path.append(path[i-1]) # add corners
            pt1 = path[i-1]
    new_path.append(path[-1]) # add end

    return new_path

def reduce_path_diag(path):
    new_path = []
    new_path.append(path[0]) # starting pos
    pt1 = path[0]
    for i in range(1, len(path)-1): 
        pt2 = path[i]
        if pt1[0] == pt2[0] or pt1[1] == pt2[1]:
            continue
        if abs(pt1[1] - pt2[1]) == abs(pt1[0] - pt2[0]): # if diagonal
            continue

        new_path.append(path[i-1]) # add corners
        pt1 = path[i-1]

    new_path.append(path[-2]) # add end
    new_path.append(path[-1]) # add end
     
    # print(f"Path Reduced from: {len(path)} to: {len(new_path)}  points by straight analysis")

    return new_path

def reduce_diagons(path):
    new_path = []
    skip_pts = []
    tol = 0.2
    look_ahead = 5 # number of points to consider on line
    # consider using a distance lookahead too

    for i in range(len(path) - look_ahead):
        pts = []
        for j in range(look_ahead):
            pts.append(path[i + j]) # generates a list of current points

        grads = []
        for j in range(1, look_ahead):
            m = f.get_gradient(pts[0], pts[j])
            grads.append(m)

        for j in range(look_ahead -2):
            ddm = abs((grads[j] - grads[-1])) / (abs(grads[-1]) + 0.0001) # div 0
            if ddm > tol: # the grad is within tolerance
                continue
            index = i + j + 1
            if index in skip_pts: # no repeats
                continue

            skip_pts.append(j + i + 1)        

    for i in range(len(path)):
        if i in skip_pts:
            continue
        new_path.append(path[i])

    print(f"Number of skipped pts: {len(skip_pts)}")

    return new_path


# externals          
def modify_path(path):
    path = reduce_path_diag(path)

    return path

    # n_pts = 10
    # new_path = []
    # for i in range(len(path)-1):
    #     new_path.append(path[i])
    #     n_pts = int(round(min(f.get_distance(path[i], path[i+1]) / 10, 1)))
    #     dx = (path[i+1][0] - path[i][0]) / n_pts
    #     dy = (path[i+1][1] - path[i][1]) / n_pts
    #     for j in range(n_pts): # add 10 points between each point
    #         x = path[i][0] + j*dx
    #         y = path[i][1] + j*dy
    #         new_path.append([x, y])
    # new_path.append(path[-1]) # last point
    # path = new_path

    # new_path = []
    # ds = 10
    # new_path.append(path[0])
    # for pt in path:
    #     dis = f.get_distance(new_path[-1], pt)
    #     if dis > ds:
    #         new_path.append(pt)

    # return new_path

