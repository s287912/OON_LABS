import random
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics as st
import copy
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from core import parameters as param
def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


# Function that adds "->" characters between the letters
def path_add_arrows(path):
    return path.replace("", "->")[2:-2]


# Get two random nodes from the node list
def sample_2_nodes(network_nodes_list):
    return random.sample(network_nodes_list, 2)


# Method to update a route_space
def update_route_space(route_space, nodes, lines):
    for path in route_space.index.tolist():
        occupancy = np.array(route_space.loc[path].to_list()[1:])
        # update occupation with switching matrix
        for i in range(len(path)-2):
            occupancy = occupancy * np.array(nodes[path[i+1]].switching_matrix[path[i]][path[i+2]])
        # update occupation with line occupation
        for i in range(len(path)-1):
            occupancy = occupancy * np.array(lines[path[i:i+2]].state)
        for channel in range(len(occupancy)):
            route_space.loc[path, "CH" + str(channel)] = occupancy[channel]
    return route_space

def plot_snr_and_bit_rate(strategy, connections):
    snr_connections = [c.snr for c in connections]
    plt.figure()
    plt.hist(snr_connections, label='SNR distribution')
    plt.title('SNR distribution with '+str(strategy)+' rate')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Number of connections')
    plt.show()

    bit_rate_connections = [c.bit_rate for c in connections if c.bit_rate != 0]
    f, ax = plt.subplots(1, 1)
    ax.hist(bit_rate_connections, label='Bit rate histogram')
    plt.title('Bit rate of accepted connections - ' + str(strategy) + ' rate')
    plt.xlabel('bit rate [bps]')
    plt.ylabel('Number of connections')
    avg_bit_rate = truncate(st.mean(bit_rate_connections) / (1e9), 3)
    tot_capacity = truncate((sum(bit_rate_connections) / (1e9)), 3)
    print('Strategy = '+str(strategy)+' rate\n'+
          "Overall average bit rates of accepted connections: ", avg_bit_rate, 'Gbps\n'+
          "Total capacity allocated into the network: " , tot_capacity, 'Gbps')
    # text
    anchored_text = AnchoredText('Average bit rate = ' + str(avg_bit_rate)+'Gbps'+ '\nTotal capacity allocated = ' + str(tot_capacity)+'Gbps', loc='upper left', pad=0.5,prop=dict(size=9))
    ax.add_artist(anchored_text)
    plt.show()


def init_traffic_matrix(network, M):
    n_nodes = len(network.nodes)
    traffic_matrix = {}
    for node1 in network.nodes.keys():
        traffic_matrix[node1] = {}
        for node2 in network.nodes.keys():
            if node1 != node2:
                traffic_matrix[node1][node2] = param.Rb_min * M
            else:
                traffic_matrix[node1][node2] = 0
    return traffic_matrix

def plot_traffic_matrix(traffic_matrix, strategy, M):
    matrix = pd.DataFrame.from_dict(traffic_matrix).to_numpy()
    nodes = list(traffic_matrix.keys())
    #print(matrix)
    fig, ax = plt.subplots()
    for r in range(0, matrix.shape[0]):
        for c in range(0, matrix.shape[1]):
            text = ax.text(c, r, matrix[r, c], ha='center', va= 'center', color='w')
    xlabel = nodes
    ylabel = nodes
    x = np.r_[:len(xlabel)]
    y = np.r_[:len(ylabel)]
    plt.title('Traffic matrix with '+str(strategy)+' rate and M= '+str(M))
    plt.xticks(x, xlabel)
    plt.yticks(y, ylabel)
    cmap = plt.cm.jet
    cmap = copy.copy(plt.cm.get_cmap("jet"))
    cmap.set_bad('orange', 1.)
    ax.imshow(matrix, interpolation='nearest',cmap=cmap)


