import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import copy
import csv
from core import utils as util
#from core import elements_lab7 as elem
from core import elements as elem
#from core import elements_git as elem2
import math
import time as time
import timeit
if __name__ == '__main__':
    start = time.time()
    N_MC = 1
    M = 1
    #net = elem.Network('resources/nodes_small.json')
    net_fixed = elem.Network('resources/nodes_full_fixed_rate.json')
    net_flex = elem.Network('resources/nodes_full_flex_rate.json')
    net_shannon = elem.Network('resources/nodes_full_shannon.json')
    #print(net.weighted_paths)
    capacity = pd.DataFrame()
    capacity['Capacity'] = []
    capacity['Rb_avg'] = []
    capacity['Rb_max'] = []
    capacity['Rb_min'] = []
    snr = pd.DataFrame()
    snr['snr_avg'] = []
    snr['snr_max'] = []
    snr['snr_min'] = []
    snr['blocking_count'] = []
    capacity_fixed = copy.deepcopy(capacity)
    capacity_flex = copy.deepcopy(capacity)
    capacity_shannon = copy.deepcopy(capacity)
    snr_fixed = copy.deepcopy(snr)
    snr_flex = copy.deepcopy(snr)
    snr_shannon = copy.deepcopy(snr)

pairs = []
for node1 in net_fixed.nodes.keys():
    for node2 in net_fixed.nodes.keys():
        if node1 != node2:
            pairs.append(str(node1+node2))
n_nodes = len(net_fixed.nodes)


net_fixed.draw

exit()
MC = True
if MC == True:
    for i in range(0, N_MC):
        connections_fixed = []
        connections_flex = []
        connections_shannon = []
        net_fixed.restore_network()
        net_flex.restore_network()
        net_shannon.restore_network()

        traffic_matrix_fixed = util.init_traffic_matrix(net_fixed, M)
        traffic_matrix_flex = util.init_traffic_matrix(net_flex, M)
        traffic_matrix_shannon = util.init_traffic_matrix(net_shannon, M)

        conn_made = n_nodes * n_nodes - n_nodes
        while conn_made > 0:
            conn_made -= net_fixed.traffic_matrix_request(traffic_matrix_fixed, connections_fixed, 1e-3, pairs)

        conn_made = n_nodes * n_nodes - n_nodes
        while conn_made > 0:
            conn_made -= net_flex.traffic_matrix_request(traffic_matrix_flex, connections_flex, 1e-3, pairs)
        conn_made = n_nodes * n_nodes - n_nodes
        while conn_made > 0:
            conn_made -= net_shannon.traffic_matrix_request(traffic_matrix_shannon, connections_shannon, 1e-3, pairs)

        capacity_fixed.loc[len(capacity_fixed.index)] = util.capacity_metrics(connections_fixed)
        snr_fixed.loc[len(snr_fixed)] = util.snr_metrics(connections_fixed)
        capacity_flex.loc[len(capacity_flex.index)] = util.capacity_metrics(connections_flex)
        snr_flex.loc[len(snr_flex)] = util.snr_metrics(connections_flex)
        capacity_shannon.loc[len(capacity_shannon.index)] = util.capacity_metrics(connections_shannon)
        snr_shannon.loc[len(snr_shannon)] = util.snr_metrics(connections_shannon)

        print((i+1)/N_MC*100, "% calculation")

    df_fixed = pd.concat([capacity_fixed, snr_fixed], axis=1, join='inner')
    df_flex = pd.concat([capacity_flex, snr_flex], axis=1, join='inner')
    df_shannon = pd.concat([capacity_shannon, snr_shannon], axis=1, join='inner')

    stop = time.time()
    print("Execution time: ", stop-start)
    print(snr_fixed)
    print(capacity_fixed)
    print(snr_flex)
    print(capacity_flex)
    print(snr_shannon)
    print(capacity_shannon)
    print("blocking event fixed rate : ", net_fixed.blocking_count)
    print("blocking event flex rate : ", net_flex.blocking_count)
    print("blocking event shannon rate : ", net_shannon.blocking_count)
    print("N. of connections fixed rate : ", len(connections_fixed))
    print("N. of connections flex rate : ", len(connections_flex))
    print("N. of connections shannon rate : ", len(connections_shannon))
    print(util.plot_traffic_matrix(traffic_matrix_shannon,'shannon', M))
    plt.show()
    df_fixed.to_csv('results/fixed_M_'+str(M)+'_MC_'+str(N_MC)+'.csv', index=False)
    df_flex.to_csv('results/flex_M_'+str(M)+'_MC_'+str(N_MC)+'.csv', index=False)
    df_shannon.to_csv('results/shannon_M_'+str(M)+'_MC_'+str(N_MC)+'.csv', index=False)

else:
    M_MAX = 20
    M = 1
    for M in range(1, M_MAX):
        connections_fixed = []
        connections_flex = []
        connections_shannon = []
        net_fixed.restore_network()
        net_flex.restore_network()
        net_shannon.restore_network()

        traffic_matrix_fixed = util.init_traffic_matrix(net_fixed, M)
        traffic_matrix_flex = util.init_traffic_matrix(net_flex, M)
        traffic_matrix_shannon = util.init_traffic_matrix(net_shannon, M)

        conn_made = n_nodes * n_nodes - n_nodes
        while conn_made > 0:
            conn_made -= net_fixed.traffic_matrix_request(traffic_matrix_fixed, connections_fixed, 1e-3, pairs)

        conn_made = n_nodes * n_nodes - n_nodes
        while conn_made > 0:
            conn_made -= net_flex.traffic_matrix_request(traffic_matrix_flex, connections_flex, 1e-3, pairs)
        conn_made = n_nodes * n_nodes - n_nodes
        while conn_made > 0:
            conn_made -= net_shannon.traffic_matrix_request(traffic_matrix_shannon, connections_shannon, 1e-3, pairs)

        capacity_fixed.loc[len(capacity_fixed.index)] = util.capacity_metrics(connections_fixed)
        snr_fixed.loc[len(snr_fixed)] = util.snr_metrics(connections_fixed)
        capacity_flex.loc[len(capacity_flex.index)] = util.capacity_metrics(connections_flex)
        snr_flex.loc[len(snr_flex)] = util.snr_metrics(connections_flex)
        capacity_shannon.loc[len(capacity_shannon.index)] = util.capacity_metrics(connections_shannon)
        snr_shannon.loc[len(snr_shannon)] = util.snr_metrics(connections_shannon)

        print((M + 1) / M_MAX * 100, "% calculation")

    df_fixed = pd.concat([capacity_fixed, snr_fixed], axis=1, join='inner')
    df_flex = pd.concat([capacity_flex, snr_flex], axis=1, join='inner')
    df_shannon = pd.concat([capacity_shannon, snr_shannon], axis=1, join='inner')

    stop = time.time()
    print("Execution time: ", stop - start)
    print(snr_fixed)
    print(capacity_fixed)
    print(snr_flex)
    print(capacity_flex)
    print(snr_shannon)
    print(capacity_shannon)
    print("blocking event fixed rate : ", net_fixed.blocking_count)
    print("blocking event flex rate : ", net_flex.blocking_count)
    print("blocking event shannon rate : ", net_shannon.blocking_count)
    print("N. of connections fixed rate : ", len(connections_fixed))
    print("N. of connections flex rate : ", len(connections_flex))
    print("N. of connections shannon rate : ", len(connections_shannon))
    print(util.plot_traffic_matrix(traffic_matrix_shannon, 'shannon', M))
    plt.show()
    exit()
    df_fixed.to_csv('results/fixed_M_MAX_' + str(M) + '.csv', index=False)
    df_flex.to_csv('results/flex_M_MAX_' + str(M) + '.csv', index=False)
    df_shannon.to_csv('results/shannon_M_MAX_' + str(M) + '.csv', index=False)
