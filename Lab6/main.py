import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand

from core import elements as elem
from core import elements_git as elem2

if __name__ == '__main__':
    net = elem.Network('resources/nodes.json')
    net2 = elem2.Network('resources/nodes.json')

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
    conn2 = elem2.Connection(node1,node2,1)
    connections.append(conn1)
    connections2.append(conn2)

#net.route_space.iloc[0] = ['occupied']*10
print(net.weighted_paths)
print(net2.weighted_paths)

net.stream(connections,'latency')
net2.stream(connections2)
print(net.weighted_paths)
#for conn in connections:
#    print(conn.input,'->',conn.output,'best snr:', conn.snr, 'best latency:', conn.latency)

#print(net.route_space)

#print(net.draw)
#print(net.weighted_paths)
stream_latency = []
stream_latency2 = []
for conn in connections:
        stream_latency.append(conn.latency)
for conn in connections2:
        stream_latency2.append(conn.latency)
print("N. conn for latency", len(stream_latency))
print("N. conn for latency2", len(stream_latency2))

plt.figure()
plt.hist(stream_latency, bins=15)
unit = " / s"
plt.xlabel('Latency'+unit)
plt.ylabel("Paths")
plt.title("Latencies")


plt.figure()
plt.hist(stream_latency2, bins=15)
unit = " / s"
plt.xlabel('Latency'+unit)
plt.ylabel("Paths")
plt.title("Latencies")
print(net.route_space)
print(net2.route_space)
plt.show()
#net.draw

