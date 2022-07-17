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
if __name__ == '__main__':
    M = 8
    net = elem.Network('resources/nodes_full_fixed_rate.json')
    print(net.weighted_paths)
    #net = elem.Network('resources/nodes_small.json')
    #net2 = elem.Network('resources/nodes_full_flex_rate.json')
    #net3 = elem.Network('resources/nodes_full_shannon.json')
pairs = []
for node1 in net.nodes.keys():
    for node2 in net.nodes.keys():
        if node1 != node2:
            pairs.append(str(node1+node2))

traffic_matrix = util.init_traffic_matrix(net, M)
n_nodes = len(net.nodes)
connections = []
conn_made = n_nodes * n_nodes #- n_nodes
while conn_made > 0:
    conn_made -= net.traffic_matrix_request(traffic_matrix, connections, 1e-3, pairs)
    #print(pd.DataFrame.from_dict(traffic_matrix).to_numpy())
    #print(traffic_matrix)


util.plot_traffic_matrix(traffic_matrix, 'fixed_rate', M)
util.wavelenght_occupation(net, M, 'fixed_rate')
plt.show()
exit()
util.plot_snr_and_bit_rate('fixed_rate', connections)

exit()

#for i in range(0, len(net.weighted_paths)):
#    print(net.weighted_paths.sort_values(by='snr', ascending=False).iloc[i],net.weighted_paths.sort_values(by='latency', ascending=True).iloc[i])

# connections = []
# connections2 = []
# connections3 = []
# for i in range(0, 10):
#     nodes = list(net.nodes.keys())
#     node1 = rand.choice(nodes)
#     nodes.remove(node1)
#     node2 = rand.choice(nodes)
# #    print("From ", node1, " to ", node2)
#     conn1 = elem.Connection(node1,node2,1e-3)
#     conn2 = elem.Connection(node1,node2,1e-3)
#     conn3 = elem.Connection(node1,node2,1e-3)
#     connections.append(conn1)
#     connections2.append(conn2)
#     connections3.append(conn3)

connections = []
with open('resources/connectionsFile.csv') as csv100ConnectionsFile:
    csvReader = csv.reader(csv100ConnectionsFile)
    for row in csvReader:
        connections.append(elem.Connection(row[0], row[1], float(row[2])))

# Saving 100 connections in a variable in order to create
# a network with not full switching matrices considering the same connections
connections2 = copy.deepcopy(connections[:])
connections3 = copy.deepcopy(connections[:])

#net.route_space.iloc[0] = ['occupied']*10

print("Prova")
net.stream(connections,'snr')
#net2.stream(connections2,'snr')
#net3.stream(connections3,'snr')
print(net.weighted_paths)

stream_snr = []
for conn in connections:
    stream_snr.append(conn.snr)
util.plot_snr_and_bit_rate('Fixed', connections)
exit()
#print(net2.weighted_paths)
#for conn in connections:
#    print(conn.input,'->',conn.output,'best snr:', conn.snr, 'best latency:', conn.latency)

#print(net.route_space)


#print(net.draw)
#print(net.weighted_paths)

stream_snr2 = []
stream_snr3 = []

for conn in connections2:
    stream_snr2.append(conn.snr)
for conn in connections3:
    stream_snr3.append(conn.snr)



util.plot_snr_and_bit_rate('Flex', connections2)
util.plot_snr_and_bit_rate('Shannon', connections3)
exit()
print(net.lines['AB'].ase_generation())

print(net.lines['AB'].n_amplifier)
print(net.lines['AB'].n_span)
print(net.weighted_paths)

#for conn in connections2:
#        stream_latency2.append(conn.snr)
print("N. conn for latency", len(stream_snr))
print("N. conn for latency2", len(stream_snr2))
f, axs = plt.subplots(1,3,sharex=True,sharey=True)
#plt.figure()
axs[0].hist(stream_snr, bins=15)
unit = " dB"
plt.xlabel('snr'+unit)
plt.ylabel("Paths")
plt.title("Snrs fixed rate")

axs[1].hist(stream_snr2, bins=15)
plt.title("Snrs flex rate")

axs[2].hist(stream_snr3, bins=15)
plt.title("Snrs shannon rate")

#plt.figure()
#plt.hist(stream_latency2, bins=15)
#unit = " dB"
#plt.xlabel('snr'+unit)
#plt.ylabel("Paths")
#plt.title("Snrs")
#print(net.route_space)
#print(net2.route_space)
plt.show()
#net.draw

