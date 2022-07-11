import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand

from core import elements as elem

if __name__ == '__main__':
    net = elem.Network('resources/nodes.json')
    net2 = elem.Network('resources/nodes.json')

#for i in range(0, len(net.weighted_paths)):
#    print(net.weighted_paths.sort_values(by='snr', ascending=False).iloc[i],net.weighted_paths.sort_values(by='latency', ascending=True).iloc[i])


connections = []
for i in range(0,100):
    nodes = list(net.nodes.keys())
    node1 = rand.choice(nodes)
    nodes.remove(node1)
    node2 = rand.choice(nodes)
#    print("From ", node1, " to ", node2)
    conn = elem.Connection(node1,node2,1)
    connections.append(conn)

#net.route_space.iloc[0] = ['occupied']*10

print("Prova")
net.stream(connections,'latency')
net2.stream(connections,'snr')
print(net.weighted_paths)
#for conn in connections:
#    print(conn.input,'->',conn.output,'best snr:', conn.snr, 'best latency:', conn.latency)

#print(net.route_space)

#print(net.draw)
#print(net.weighted_paths)
stream_snr = []
stream_latency = []
for conn in connections:
        if conn.latency != 0:
            stream_latency.append(conn.latency)
        if conn.snr != 0:
            stream_snr.append(conn.snr)
print("N. conn for snr", len(stream_snr))
print("N. conn for latency", len(stream_latency))
plt.figure()
plt.hist(stream_latency, bins=15)
unit = " / s"
plt.xlabel('Latency'+unit)
plt.ylabel("Paths")
plt.title("Latencies")


plt.figure()
plt.hist(stream_snr, bins=15)
unit = " / s"
plt.xlabel('Snr'+unit)
plt.ylabel("Paths")
plt.title("SNRs")
#plt.show()
#net.draw

print(net.lines['AB'].state)