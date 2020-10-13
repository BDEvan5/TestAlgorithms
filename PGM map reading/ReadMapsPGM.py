import yaml
import numpy as np
from matplotlib import pyplot as plt


def read_map(map_name="columbia"):
    map_name = 'PGM map reading/maps/' + map_name 
    yaml_file = read_yaml_file(map_name + '.yaml')

    map_file_name = yaml_file['image']
    free_thresh = yaml_file['free_thresh']

    pgm_name = 'PGM map reading/maps/' + map_file_name
    map_data = data = readpgm(pgm_name)

    show_map(map_data)



def show_map(map_arr):
    plt.figure(1)
    plt.imshow(map_arr)

    plt.show()

def read_yaml_file(file_name, print_out=False):
    with open(file_name) as file:
        documents = yaml.full_load(file)

        yaml_file = documents.items()
        if print_out:
            for item, doc in yaml_file:
                print(item, ":", doc)

    yaml_file = dict(yaml_file)
    return yaml_file


def readpgm(name):
    with open(name) as f:
        lines = f.readlines()

    # This ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
    # here,it makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    data = (np.array(data[3:]),(data[1],data[0]),data[2])
    data = np.reshape(data[0],data[1])

    return data 


if __name__ == "__main__":
    read_map()
    # testing()
