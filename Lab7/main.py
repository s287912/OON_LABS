import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand

from core import elements_v1 as elem
from core import elements_old as elem2

if __name__ == '__main__':
    net = elem.Network('resources/nodes_full.json')
    net2 = elem.Network('resources/nodes_not_full.json')

#for i in range(0, len(net.weighted_paths)):
#    print(net.weighted_paths.sort_values(by='snr', ascending=False).iloc[i],net.weighted_paths.sort_values(by='latency', ascending=True).iloc[i])

connections = []
connections2 = []
for i in range(0, 10):
    nodes = list(net.nodes.keys())
    node1 = rand.choice(nodes)
    nodes.remove(node1)
    node2 = rand.choice(nodes)
#    print("From ", node1, " to ", node2)
    conn1 = elem.Connection(node1,node2,1)
    conn2 = elem.Connection(node1,node2,1)
    connections.append(conn1)
    connections2.append(conn2)

#net.route_space.iloc[0] = ['occupied']*10

print("Prova")
net.stream(connections,'snr')
net2.stream(connections2,'snr')
print(net.weighted_paths)
#for conn in connections:
#    print(conn.input,'->',conn.output,'best snr:', conn.snr, 'best latency:', conn.latency)

#print(net.route_space)

#print(net.draw)
#print(net.weighted_paths)
stream_snr = []
stream_snr2 = []
for conn in connections:
        stream_snr.append(conn.snr)

for conn in connections2:
        stream_snr2.append(conn.snr)
print("N. conn for snr", len(stream_snr))
print("N. conn for snr", len(stream_snr2))

plt.figure()
plt.hist(stream_snr, bins=15)
unit = " / s"
plt.xlabel('Snr'+unit)
plt.ylabel("Paths")
plt.title("SNRs with fully connected")


plt.figure()
plt.hist(stream_snr2, bins=15)
unit = " / s"
plt.xlabel('Snr'+unit)
plt.ylabel("Paths")
plt.title("SNRs with not fully connected")

#plt.show()
#net.draw

